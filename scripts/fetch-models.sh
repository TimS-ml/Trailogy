#!/usr/bin/env bash
# Downloads the Kokoro MLX model files from mlalma's KokoroTestApp Git LFS
# storage. We bypass git-lfs install entirely by hitting GitHub's raw URL —
# GitHub auto-redirects raw.githubusercontent.com requests for LFS-tracked
# objects to the LFS storage host.
#
# Files (placed in HikeCompanion/Resources/Models/):
#   kokoro-v1_0.safetensors  ~600 MB   neural network weights
#   voices.npz               ~30 MB    28 voice embeddings
#
# Usage:  bash scripts/fetch-models.sh
# Re-run safe: skips files that already exist with non-trivial size.

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/HikeCompanion/Resources/Models"
RAW="https://github.com/mlalma/KokoroTestApp/raw/main/Resources"

mkdir -p "$DEST"

download() {
  local file="$1"
  local out="$DEST/$file"
  if [[ -f "$out" ]]; then
    local sz
    sz=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000000 ]]; then
      echo "  $file already present ($(du -h "$out" | cut -f1)) — skipping"
      return 0
    fi
    echo "  $file exists but is suspiciously small ($sz bytes); re-downloading"
    rm -f "$out"
  fi
  echo "==> Downloading $file ..."
  curl -fL --progress-bar "$RAW/$file" -o "$out"
}

download "kokoro-v1_0.safetensors"
download "voices.npz"

echo ""
echo "==> Done."
ls -lh "$DEST" | grep -v '^total'
echo ""
echo "Total Models/ size: $(du -sh "$DEST" | cut -f1)"
