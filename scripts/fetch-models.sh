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

# ---------------------------------------------------------------------------
# MiniLM (RAG embedder) — sentence-transformers/all-MiniLM-L6-v2
# ---------------------------------------------------------------------------
# Bundled into Models/MiniLM/ so RAGService loads it from app resources
# (no first-run HuggingFace download required). ~87 MB total.
# Pulls from huggingface.co's resolve/main URLs which serve LFS through
# their CDN — no `git-lfs` install or auth needed for this public model.

MINILM_DEST="$DEST/MiniLM"
MINILM_BASE="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"
mkdir -p "$MINILM_DEST"

download_minilm() {
  local file="$1"
  local min_size="${2:-100}"
  local out="$MINILM_DEST/$file"
  if [[ -f "$out" ]]; then
    local sz
    sz=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -gt "$min_size" ]]; then
      echo "  MiniLM/$file already present ($(du -h "$out" | cut -f1)) — skipping"
      return 0
    fi
    rm -f "$out"
  fi
  echo "==> Downloading MiniLM/$file ..."
  curl -fL --progress-bar "$MINILM_BASE/$file" -o "$out"
}

download_minilm "model.safetensors"     1000000
download_minilm "config.json"           100
download_minilm "tokenizer.json"        10000
download_minilm "tokenizer_config.json" 100
download_minilm "special_tokens_map.json" 50
download_minilm "vocab.txt"             10000

echo ""
echo "==> Done."
ls -lh "$DEST" | grep -v '^total'
if [[ -d "$MINILM_DEST" ]]; then
  echo ""
  echo "MiniLM:"
  ls -lh "$MINILM_DEST" | grep -v '^total'
fi
echo ""
echo "Total Models/ size: $(du -sh "$DEST" | cut -f1)"
