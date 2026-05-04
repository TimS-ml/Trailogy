#!/usr/bin/env bash
# Downloads Gemma 4 E2B (INT4 MLX format) from Hugging Face into the
# bundle resource directory. ~3.5 GB total.
#
# Primary repo:   mlx-community/gemma-4-e2b-it-4bit  (Apache 2.0)
# Backup repo:    unsloth/gemma-4-E2B-it-UD-MLX-4bit (if primary has the
#                 PLE quantization bug — pass --backup to use this)
#
# After this script finishes, re-run scripts/generate-project.sh and rebuild
# in Xcode so the new files get bundled.
#
# Usage:
#   bash scripts/fetch-gemma.sh             # primary repo
#   bash scripts/fetch-gemma.sh --backup    # unsloth fallback
#
# Re-run safe: skips files that already exist with non-trivial size.

set -euo pipefail

REPO="mlx-community/gemma-4-e2b-it-4bit"
if [[ "${1:-}" == "--backup" ]]; then
  REPO="unsloth/gemma-4-E2B-it-UD-MLX-4bit"
  echo "==> Using backup repo: $REPO"
fi

DEST="$(cd "$(dirname "$0")/.." && pwd)/HikeCompanion/Resources/Models/Gemma"
API="https://huggingface.co/api/models/${REPO}/tree/main"
RESOLVE="https://huggingface.co/${REPO}/resolve/main"

mkdir -p "$DEST"

echo "==> Listing files at $REPO ..."
LIST=$(curl -fsSL "$API")
FILES=$(echo "$LIST" | python3 -c '
import json, sys
items = json.load(sys.stdin)
# Only top-level files; MLX models are flat (no nested dirs)
for it in items:
    if it.get("type") == "file":
        print(it["path"])
')

if [[ -z "$FILES" ]]; then
  echo "ERROR: HF API returned no files for $REPO" >&2
  exit 1
fi

echo "==> $(echo "$FILES" | wc -l | tr -d ' ') files to fetch"

while IFS= read -r file; do
  [[ -z "$file" ]] && continue
  out="$DEST/$file"
  if [[ -f "$out" ]]; then
    sz=$(stat -f %z "$out" 2>/dev/null || stat -c %s "$out" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000 ]]; then
      echo "  skip $file ($(du -h "$out" | cut -f1))"
      continue
    fi
  fi
  echo "==> $file"
  mkdir -p "$(dirname "$out")"
  curl -fL --progress-bar "$RESOLVE/$file" -o "$out"
done <<< "$FILES"

echo ""
echo "==> Hoisting Gemma processor fields to top level for mlx-swift-lm..."
# Why (hoist): HF's official Gemma 4 processor_config.json keeps
# image-processor fields nested under "image_processor" (size,
# image_mean, image_std, do_normalize). mlx-swift-lm's
# Gemma4ProcessorConfiguration decoder expects them flat at the top
# level. Without this hoist the Swift decoder gets nil for `size` and
# falls back to 800x800, which blows up vision-tower memory.
#
# Why (size override to 960x672 instead of upstream 224x224):
# Gemma 4's trained vision pooler is a kernel=3 stride pooler over
# the 14×20 (or 20×14) bucket grid that exactly fills the
# max_patches=2520 budget. The mlx-swift-lm pooler degenerates at
# 224x224 (14×14=196 patches → 196 raw + 84 zero outputs instead of
# the trained 280 pooled features). 960x672 produces 60×42=2520
# patches → 20×14=280 cleanly-pooled tokens, matching what the
# language model was trained to attend to.
#
# We do NOT modify the architectural fields (image_seq_length=280,
# pooling_kernel_size=3, default_output_length=280) — they are trained.
# The increased-memory-limit entitlement absorbs the 2520-patch
# encoder pass.
TRAINED_SIZE='{"height": 960, "width": 672}'
PCFG="$DEST/processor_config.json"
if [[ -f "$PCFG" ]]; then
  python3 - "$PCFG" "$TRAINED_SIZE" <<'PY'
import json, sys
p = sys.argv[1]
trained_size = json.loads(sys.argv[2])
with open(p) as f:
    cfg = json.load(f)
ip = cfg.get("image_processor", {})
# Top-level keys mlx-swift-lm's Gemma4ProcessorConfiguration reads.
# We HOIST mean/std/do_normalize from image_processor (they're correct
# for Gemma 4 there), but FORCE size to the trained 960×672 shape.
patch = {
    "do_normalize": ip.get("do_normalize", False),
    "image_mean":   ip.get("image_mean",   [0.0, 0.0, 0.0]),
    "image_std":    ip.get("image_std",    [1.0, 1.0, 1.0]),
    "size":         trained_size,
}
changed = []
for k, v in patch.items():
    if cfg.get(k) != v:
        cfg[k] = v
        changed.append(k)
# Also override the nested image_processor.size so HF tooling agrees.
if ip and ip.get("size") != trained_size:
    ip["size"] = trained_size
    changed.append("image_processor.size")
if changed:
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    print("   patched: " + ", ".join(changed))
else:
    print("   already patched (no-op)")
PY
fi

echo ""
echo "==> Done. Models/Gemma/ contents:"
ls -lh "$DEST" | grep -v '^total'
echo ""
echo "Total Gemma/ size: $(du -sh "$DEST" | cut -f1)"
echo ""
echo "Next: bash scripts/generate-project.sh   (so Xcode picks up the new files)"
