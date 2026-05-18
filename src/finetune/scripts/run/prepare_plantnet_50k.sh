#!/usr/bin/env bash
# prepare_plantnet_50k.sh — Reproducible PlantNet-300K 50k SFT data prep.
#
# Purpose
# -------
# Produce a CANONICAL set of JSONL files (train / val / test) that are
# byte-consistent across machines (macOS, Linux GPU box, remote cloud)
# so training and eval consume the same data everywhere. This is the
# single source of truth for the 50k SFT dataset used by the kept
# training configs under ``configs/``.
#
# Data source
# -----------
# PlantNet-300K v2 (Garcin et al., NeurIPS Datasets and Benchmarks 2021):
#   https://zenodo.org/records/10419064
#
# Download the archive and extract so the resulting directory has the
# layout below; the default PLANTNET_ROOT below assumes this directory
# sits next to (i.e. as a sibling of) the App repo root.
#
#   PlantNet-300K-data-v2/
#     plantnet300K_metadata.csv
#     species_metadata_enriched.csv         (will be created from repo
#                                            assets/ if missing — see
#                                            "Enriched CSV" section)
#     train/<sid>/<hash>.jpg
#     val/<sid>/<hash>.jpg
#     test/<sid>/<hash>.jpg
#
# Output layout (under $OUTPUT_DIR)
# ---------------------------------
#   train.jsonl   ~45k rows. English vernacular + Wikipedia description
#                 prompts. Sampled from PlantNet-300K-data-v2/train/.
#   val.jsonl     ~5k rows. PER-SPECIES STRATIFIED holdout of the same
#                 50k pool. Every species in train.jsonl with >=2 imgs
#                 in the pool also appears in val.jsonl (no orphan
#                 species). Same data-generating distribution as
#                 train.jsonl (NOT the official test split — do not
#                 report deliverable numbers on this). v1 was
#                 `samples[:val_count]` random slice which left ~23 % of
#                 species missing from val; v2 carves per species.
#   test.jsonl    ~30k rows. Drawn from PlantNet-300K-data-v2/test/
#                 (paper-grade out-of-distribution). Use THIS file for
#                 quantization eval and any number that goes into the
#                 report.
#   train_legacy_random.jsonl  (only if a v1-era file existed before
#                 this script first ran) — the historical random-slice
#                 outputs, kept so prior eval numbers remain reproducible.
#   val_legacy_random.jsonl
#   test_legacy_random.jsonl
#   images_resized/
#     train/<sid>/<hash>.jpg    pre-resized to 960x672 (iOS runtime shape)
#     val/<sid>/<hash>.jpg
#     test/<sid>/<hash>.jpg
#   filter_report.json          species-filter stats (train+val pass)
#   filter_report_test.json     species-filter stats (test pass)
#
# train/val are in-distribution to each other (same train/ pool). test
# is out-of-distribution to both. This also avoids the old
# val.jsonl-from-test/ vs val_mac.jsonl-from-train/
# mismatch: this script eliminates that mismatch by always emitting all
# three files together from a single deterministic invocation.
#
# Enriched CSV
# ------------
# species_metadata_enriched.csv carries the English common-name +
# Wikipedia-description metadata. We ship a frozen copy at
# <repo>/assets/species_metadata_enriched.csv so external reproducers
# don't need to regenerate it from GBIF + Wikipedia. If
# $PLANTNET_ROOT/species_metadata_enriched.csv is missing, this script
# copies the repo's frozen copy in.
#
# Configurable env vars (defaults in brackets)
# --------------------------------------------
#   PLANTNET_ROOT  [<repo>/../PlantNet-300K-data-v2]
#                  Absolute or relative path to the PlantNet-300K v2 dir.
#   OUTPUT_DIR     [<repo>/src/finetune/data/english-desc]
#                  Where train.jsonl / val.jsonl / test.jsonl land.
#   SEED           [42]
#   MAX_SAMPLES    [50000]   Cap for the train+val pool (drawn from train/).
#   VAL_RATIO      [0.1]     In-distribution holdout fraction.
#   RESIZE         [960x672] HxW to pre-stretch to (matches iOS).
#                            Pass "none" to skip resize (paths point at
#                            originals — NOT recommended for prod).
#   PYTHON         [python]  Python binary. Override if you need a
#                            specific conda env, e.g. PYTHON=/path/to/python.
#
# Usage
# -----
#   # 1. Default: PlantNet-300K-data-v2 sits next to the repo root
#   bash src/finetune/scripts/run/prepare_plantnet_50k.sh
#
#   # 2. Custom data root
#   PLANTNET_ROOT=/data/plantnet300k_v2 \
#     bash src/finetune/scripts/run/prepare_plantnet_50k.sh
#
#   # 3. Different python env
#   PYTHON=/path/to/python \
#     bash src/finetune/scripts/run/prepare_plantnet_50k.sh
#
# Determinism
# -----------
# All randomness inside prepare_plantnet_enriched.py is seeded by --seed.
# The train/val/test split, the per-species stratified sampling, and the
# per-record prompt-template selection all derive from random.Random(SEED).
# Two invocations of this script on different machines with the same
# PlantNet-300K-data-v2 input produce byte-identical JSONL files (up to
# absolute path differences in the "image" field, which we keep as
# absolute to match the existing convention).
#
# Run wall time (single-GPU desktop, NVMe disk): ~3-5 min for resize+JSONL emit.

set -euo pipefail

# --- Path resolution ---------------------------------------------------
# This script lives at:
#   <repo>/src/finetune/scripts/run/prepare_plantnet_50k.sh
# Resolve key roots relative to its own location so it works regardless
# of cwd.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"           # .../src/finetune
SRC_DIR="$(cd "$FINETUNE_DIR/.." && pwd)"                 # .../src
APP_DIR="$(cd "$SRC_DIR/.." && pwd)"                      # .../<repo>
HC_DIR="$APP_DIR"                                         # back-compat alias

# --- Configuration -----------------------------------------------------

PLANTNET_ROOT_DEFAULT="$APP_DIR/PlantNet-300K-data-v2"

PLANTNET_ROOT="${PLANTNET_ROOT:-$PLANTNET_ROOT_DEFAULT}"
OUTPUT_DIR="${OUTPUT_DIR:-$FINETUNE_DIR/data/english-desc}"
SEED="${SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-50000}"
VAL_RATIO="${VAL_RATIO:-0.1}"
RESIZE="${RESIZE:-960x672}"
PYTHON="${PYTHON:-python}"

# Resolve PLANTNET_ROOT and OUTPUT_DIR to absolute paths so the JSONL
# image fields are stable absolute paths (existing convention).
PLANTNET_ROOT="$(cd "$PLANTNET_ROOT" 2>/dev/null && pwd || echo "$PLANTNET_ROOT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"

echo "=== prepare_plantnet_50k.sh ==="
echo "  PLANTNET_ROOT : $PLANTNET_ROOT"
echo "  OUTPUT_DIR    : $OUTPUT_DIR"
echo "  SEED          : $SEED"
echo "  MAX_SAMPLES   : $MAX_SAMPLES  (train+val pool, drawn from train/)"
echo "  VAL_RATIO     : $VAL_RATIO    (in-distribution holdout fraction)"
echo "  RESIZE        : $RESIZE"
echo "  PYTHON        : $PYTHON"
echo ""

# --- Validation --------------------------------------------------------

if [ ! -d "$PLANTNET_ROOT" ]; then
    echo "ERROR: PLANTNET_ROOT does not exist: $PLANTNET_ROOT" >&2
    echo "" >&2
    echo "Download PlantNet-300K v2 from:" >&2
    echo "  https://zenodo.org/records/10419064" >&2
    echo "" >&2
    echo "Extract so the directory has train/, val/, test/ subdirs, then" >&2
    echo "place it next to this repo (or set PLANTNET_ROOT explicitly)." >&2
    exit 1
fi

for sub in train test; do
    if [ ! -d "$PLANTNET_ROOT/$sub" ]; then
        echo "ERROR: missing required split: $PLANTNET_ROOT/$sub" >&2
        echo "Re-extract the Zenodo archive — it should contain train/, val/, test/." >&2
        exit 1
    fi
done

# --- Enriched CSV provisioning ----------------------------------------
# species_metadata_enriched.csv lives at:
#   - PlantNet-300K-data-v2/species_metadata_enriched.csv  (canonical, in-place)
#   - <repo>/assets/species_metadata_enriched.csv          (repo-shipped copy)
# If the in-place one is missing, copy the repo copy into PlantNet-300K-data-v2/.
# These two are kept byte-identical; the repo copy is the source of truth
# for fresh PlantNet downloads.

ENRICH_CSV="$PLANTNET_ROOT/species_metadata_enriched.csv"
REPO_ENRICH_CSV="$HC_DIR/assets/species_metadata_enriched.csv"

if [ ! -f "$ENRICH_CSV" ]; then
    if [ ! -f "$REPO_ENRICH_CSV" ]; then
        echo "ERROR: enriched CSV not found at either of:" >&2
        echo "  $ENRICH_CSV"     >&2
        echo "  $REPO_ENRICH_CSV" >&2
        echo "" >&2
        echo "The repo copy at <repo>/assets/ should always be" >&2
        echo "present. If you've deleted it, restore from git." >&2
        exit 1
    fi
    echo "species_metadata_enriched.csv missing in PlantNet root — copying from repo assets/..."
    cp "$REPO_ENRICH_CSV" "$ENRICH_CSV"
    echo "  → $ENRICH_CSV"
    echo ""
fi

# --- Step 0: snapshot existing train/val/test as *_legacy_random.jsonl ----
# v2 of the val split (per-species stratified, replaces v1's random
# samples[:val_count] slice) produces a structurally different val.jsonl
# that's NOT comparable to historic eval numbers gathered against the old
# val. Keep the old artifacts under *_legacy_random.jsonl so prior results
# remain reproducible — but only on the FIRST re-prep (don't overwrite a
# pre-existing legacy file from a still-earlier prep).
#
# Idempotent: if legacy already exists, skip the rename. If the source
# file is missing (fresh checkout), skip too.

snapshot_legacy() {
    local src="$1"
    local legacy="$2"
    if [ -f "$legacy" ]; then
        echo "  legacy snapshot already exists, keeping it: $legacy"
        return 0
    fi
    if [ ! -f "$src" ]; then
        echo "  no source to snapshot: $src (fresh checkout)"
        return 0
    fi
    mv "$src" "$legacy"
    echo "  $src → $legacy"
}

echo "=== Step 0: snapshot existing JSONL as *_legacy_random.jsonl ==="
snapshot_legacy "$OUTPUT_DIR/train.jsonl"  "$OUTPUT_DIR/train_legacy_random.jsonl"
snapshot_legacy "$OUTPUT_DIR/val.jsonl"    "$OUTPUT_DIR/val_legacy_random.jsonl"
snapshot_legacy "$OUTPUT_DIR/test.jsonl"   "$OUTPUT_DIR/test_legacy_random.jsonl"
echo ""

# --- Step 1: train + val (per-species stratified split from train/) -----

echo "=== Step 1: train.jsonl + val.jsonl  (per-species split of PlantNet train/) ==="
(
    cd "$FINETUNE_DIR"
    "$PYTHON" -m src.prepare_plantnet_enriched \
        --plantnet_root  "$PLANTNET_ROOT" \
        --enriched_csv   "$ENRICH_CSV" \
        --data_version   v2 \
        --split          train \
        --output_dir     "$OUTPUT_DIR" \
        --max_samples    "$MAX_SAMPLES" \
        --val_ratio      "$VAL_RATIO" \
        --resize_to      "$RESIZE" \
        --seed           "$SEED"
)

echo ""

# --- Step 2: test.jsonl  (from PlantNet test/, all → val.jsonl in stage)
# prepare_plantnet_enriched.py writes to <output_dir>/{train,val}.jsonl
# and resized images to <output_dir>/images_resized/{train,val}/ based on
# the train/val split — NOT based on the --split arg (the for-loop in
# main() iterates ("train", train_samples) and ("val", val_samples)).
# So to produce a test.jsonl from --split test, we:
#   1. Stage the run in a sibling dir with --val_ratio 1.0 so EVERY
#      sample lands in the "val" bucket inside the staging dir.
#   2. Promote the staged val.jsonl → $OUTPUT_DIR/test.jsonl and the
#      staged images_resized/val/ → $OUTPUT_DIR/images_resized/test/.
#   3. Rewrite the absolute paths inside test.jsonl from the staging
#      location to the final location.

STAGE_DIR="$OUTPUT_DIR/_test_stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

# Count test/ jpgs to size the --max_samples cap so all images are kept.
TEST_IMG_COUNT="$(find "$PLANTNET_ROOT/test" -type f -name '*.jpg' | wc -l)"
TEST_MAX_SAMPLES="$TEST_IMG_COUNT"

echo "=== Step 2: test.jsonl  (from PlantNet test/, $TEST_IMG_COUNT jpgs) ==="
(
    cd "$FINETUNE_DIR"
    "$PYTHON" -m src.prepare_plantnet_enriched \
        --plantnet_root  "$PLANTNET_ROOT" \
        --enriched_csv   "$ENRICH_CSV" \
        --data_version   v2 \
        --split          test \
        --output_dir     "$STAGE_DIR" \
        --max_samples    "$TEST_MAX_SAMPLES" \
        --val_ratio      1.0 \
        --resize_to      "$RESIZE" \
        --seed           "$SEED"
)

echo ""
echo "=== Step 3: promote staged outputs → test.jsonl + images_resized/test/ ==="

mkdir -p "$OUTPUT_DIR/images_resized"

# Move resized test images into the final layout. Stage emitted them
# under images_resized/val/ because val_ratio=1.0.
if [ -d "$STAGE_DIR/images_resized/val" ]; then
    rm -rf "$OUTPUT_DIR/images_resized/test"
    mv "$STAGE_DIR/images_resized/val" "$OUTPUT_DIR/images_resized/test"
fi

# Rewrite absolute image paths inside the staged val.jsonl from the
# staging location to the final location, then save as test.jsonl.
#
# Both the JSONL paths and the stage/target prefixes are passed
# through ``os.path.realpath`` before the .replace() call. The inner
# python script (prepare_plantnet_enriched.py) emits ``str(img_path.resolve())``
# which follows symlinks, so a directory layout like
# ``data/english-desc -> english-desc-v2`` makes the JSONL record paths
# rooted at the realpath even though ``$STAGE_DIR`` / ``$OUTPUT_DIR``
# (passed in from bash) keep the symlink form. Without normalizing
# both sides, the replace would silently miss and the resulting
# test.jsonl would point at the staging dir that this script deletes
# at the end. Verified on a Linux 4090 box where the symlink existed;
# safe no-op on filesystems with no symlinks (realpath==input).
"$PYTHON" - <<PYEOF
import json
import os
import pathlib
import sys

stage_jsonl  = pathlib.Path("${STAGE_DIR}/val.jsonl")
target_jsonl = pathlib.Path("${OUTPUT_DIR}/test.jsonl")
stage_str    = os.path.realpath("${STAGE_DIR}/images_resized/val") + "/"
target_str   = os.path.realpath("${OUTPUT_DIR}/images_resized/test") + "/"

if not stage_jsonl.exists():
    sys.exit(f"ERROR: staged JSONL not found: {stage_jsonl}")

n = 0
n_rewritten = 0
with stage_jsonl.open() as fin, target_jsonl.open("w") as fout:
    for line in fin:
        rec = json.loads(line)
        original = rec["image"]
        # Normalize the record's image path before substitution so the
        # match works regardless of which prefix form the inner script
        # emitted (resolved or symlinked).
        rec["image"] = os.path.realpath(original).replace(stage_str, target_str)
        if rec["image"] != original:
            n_rewritten += 1
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1

print(f"  Wrote {n} records → {target_jsonl} ({n_rewritten} path-rewritten)")
if n_rewritten == 0:
    sys.exit(
        f"ERROR: no image paths were rewritten from {stage_str!r} to {target_str!r}.\n"
        "The staged val.jsonl emit format may have changed; inspect a record:\n"
        f"  head -1 {stage_jsonl}"
    )
PYEOF

# Preserve the test-side filter_report under a distinct name (the
# train-side one already lives at $OUTPUT_DIR/filter_report.json).
if [ -f "$STAGE_DIR/filter_report.json" ]; then
    mv "$STAGE_DIR/filter_report.json" "$OUTPUT_DIR/filter_report_test.json"
fi

# Stage cleanup.
rm -rf "$STAGE_DIR"

# --- Summary -----------------------------------------------------------

echo ""
echo "=== Summary ==="
for f in train.jsonl val.jsonl test.jsonl; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        printf "  %-12s %6d rows\n" "$f" "$(wc -l < "$OUTPUT_DIR/$f")"
    else
        printf "  %-12s MISSING\n" "$f"
    fi
done
echo "  Output dir   : $OUTPUT_DIR"
echo "  Reports      : $OUTPUT_DIR/filter_report.json"
echo "                 $OUTPUT_DIR/filter_report_test.json"
echo ""
echo "Done."
