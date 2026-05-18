#!/bin/bash
# Export finetuned LoRA adapter to MLX format for the iOS app.
#
# Two workflows:
#
# A) Full export on a Mac with both NVIDIA-trained adapter and MLX installed:
#    bash scripts/run/export.sh outputs/hike-gemma4-lora/final-adapter
#
# B) Split workflow (merge on NVIDIA box, convert on Mac):
#    # On NVIDIA box:
#    python src/export_mlx.py --adapter_path outputs/.../final-adapter --merge_only
#    # Transfer merged_model/ to Mac, then:
#    python src/export_mlx.py --merged_dir merged_model/ --output_dir mlx_export/

set -euo pipefail
cd "$(dirname "$0")/../.."

ADAPTER_PATH="${1:-outputs/hike-gemma4-lora/final-adapter}"
OUTPUT_DIR="${2:-mlx_export}"
QUANT_BITS="${3:-4}"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter not found at $ADAPTER_PATH"
    echo "Usage: bash scripts/run/export.sh <adapter_path> [output_dir] [quant_bits]"
    exit 1
fi

echo "=== Export to MLX ==="
echo "Adapter:  $ADAPTER_PATH"
echo "Output:   $OUTPUT_DIR"
echo "Quantize: ${QUANT_BITS}-bit"
echo ""

python src/export_mlx.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --quantize_bits "$QUANT_BITS" \
    --strip_audio

echo ""
echo "=== Export complete ==="
echo "MLX model at: $OUTPUT_DIR"
echo ""
echo "To deploy to the iOS app:"
echo "  cp $OUTPUT_DIR/*.safetensors path/to/hikeCompanion/HikeCompanion/Resources/Models/Gemma/"
echo "  cp $OUTPUT_DIR/config.json   path/to/hikeCompanion/HikeCompanion/Resources/Models/Gemma/"
