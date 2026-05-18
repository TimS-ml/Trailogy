#!/usr/bin/env python3
"""[RUNTIME] Mac (Apple Silicon, M-series) — splice mlx-lm LM-only quant into a multimodal dir.

Takes an LM-only quantized output produced by ``mlx_lm.quant.*`` (e.g.
``quantization/results/mac_mlx_lm/dynamic-quant-bpw4/``) and re-attaches
the BF16 ``vision_tower.*`` + ``embed_vision.*`` weights from the
original merged BF16 dir, producing an mlx_vlm-loadable multimodal
checkpoint suitable for ``run_eval.py --loader mlx_vlm``.

Key format conventions:

    HF (in bf16 merged dir)            mlx_vlm (target)
    ---------------------------------  -------------------------------
    model.vision_tower.*               vision_tower.*
    model.embed_vision.*               embed_vision.*
    model.language_model.*             language_model.model.*
    (audio_tower stripped — iOS doesn't use it)

We DO NOT touch the LM keys — they come from the mlx-lm-native quant
output and are already in the correct mlx_vlm format.

Usage::

    python -m scripts.repair.splice_lm_into_multimodal \\
        --lm-dir      quantization/results/mac_mlx_lm/dynamic-quant-bpw4 \\
        --bf16-dir    ~/work/gemma4-merged-bf16 \\
        --output-dir  quantization/results/mac_mlx_lm/dynamic-quant-bpw4_spliced \\
        [--include-audio]   # default: strip audio_tower + embed_audio
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("splice")


def read_st_header(path: Path) -> tuple[dict, int]:
    """Return (tensor_index, header_end_offset)."""
    with open(path, "rb") as f:
        (hdr_len,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hdr_len))
    return hdr, 8 + hdr_len


def transform_bf16_key(k: str) -> str | None:
    """HF key naming -> mlx_vlm key naming. Returns None to skip."""
    if not k.startswith("model."):
        return None
    inner = k[len("model."):]
    top = inner.split(".", 1)[0]
    if top in ("vision_tower", "embed_vision"):
        return inner  # strip leading "model."
    if top in ("audio_tower", "embed_audio"):
        return None   # let caller include or skip
    if top == "language_model":
        # We don't include LM from bf16 — caller uses the LM-only quant.
        return None
    # Anything else (e.g. lm_head if it lived at model.lm_head) — pass through.
    return inner


def splice(
    lm_dir: Path,
    bf16_dir: Path,
    output_dir: Path,
    include_audio: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Read both safetensors headers (no weight load yet) ---
    lm_st = lm_dir / "model.safetensors"
    bf16_st = bf16_dir / "model.safetensors"
    assert lm_st.exists(), f"missing {lm_st}"
    assert bf16_st.exists(), f"missing {bf16_st}"

    lm_hdr, _ = read_st_header(lm_st)
    bf16_hdr, _ = read_st_header(bf16_st)
    lm_keys = [k for k in lm_hdr if k != "__metadata__"]
    bf16_keys = [k for k in bf16_hdr if k != "__metadata__"]

    log.info("LM-only dir: %d tensors", len(lm_keys))
    log.info("BF16 dir:    %d tensors", len(bf16_keys))

    # --- 2. Determine which bf16 keys to bring in, with renaming ---
    bf16_keep: list[tuple[str, str]] = []  # (bf16_key, new_mlx_key)
    for k in bf16_keys:
        new_k = transform_bf16_key(k)
        if new_k is None:
            top = k.split(".", 1)[1].split(".", 1)[0] if k.startswith("model.") else "?"
            if include_audio and top in ("audio_tower", "embed_audio"):
                new_k = k[len("model."):]
            else:
                continue
        bf16_keep.append((k, new_k))

    log.info("Will bring in %d bf16 tensors (audio %s)",
             len(bf16_keep), "included" if include_audio else "stripped")

    # Check for collisions: LM-only output shouldn't already define
    # vision/audio keys, but verify.
    new_keys_to_add = {nk for _, nk in bf16_keep}
    collisions = new_keys_to_add & set(lm_keys)
    if collisions:
        raise ValueError(
            f"Key collision between LM-only dir and bf16 vision keys: "
            f"{sorted(collisions)[:5]}... (total {len(collisions)})"
        )

    # --- 3. Use MLX safetensors loader/saver (handles bf16 natively;
    #         numpy can't) ---
    import mlx.core as mx

    merged: dict[str, mx.array] = {}

    # mx.load returns a dict[str, mx.array] when given a single safetensors path.
    lm_tree = mx.load(str(lm_st))
    for k in lm_keys:
        merged[k] = lm_tree[k]
    log.info("Loaded %d LM tensors via mx.load", len(lm_keys))

    bf16_tree = mx.load(str(bf16_st))
    for orig_k, new_k in bf16_keep:
        merged[new_k] = bf16_tree[orig_k]
    log.info("Loaded %d bf16 vision tensors via mx.load", len(bf16_keep))

    # --- 4. Write merged safetensors ---
    out_st = output_dir / "model.safetensors"
    mx.save_safetensors(str(out_st), merged, metadata={"format": "mlx"})
    out_size = out_st.stat().st_size
    log.info("Wrote %s (%.3f GB)", out_st, out_size / 1e9)

    # --- 5. Build merged config.json ---
    # Start from LM-only config (carries quantization metadata + text_config).
    lm_cfg = json.loads((lm_dir / "config.json").read_text())
    bf16_cfg = json.loads((bf16_dir / "config.json").read_text())

    # Bring vision_config (and audio_config if including audio) from bf16.
    if "vision_config" in bf16_cfg:
        lm_cfg["vision_config"] = bf16_cfg["vision_config"]
    if include_audio and "audio_config" in bf16_cfg:
        lm_cfg["audio_config"] = bf16_cfg["audio_config"]
    elif "audio_config" in lm_cfg and not include_audio:
        # Drop audio_config if we're not shipping audio weights, to avoid
        # mlx_vlm trying to instantiate an audio encoder it can't fill.
        # (Keep it actually — mlx_vlm sanitize is forgiving; removing
        # might confuse model_type dispatch.)
        pass

    # Make sure architectures + model_type say multimodal.
    lm_cfg["architectures"] = ["Gemma4ForConditionalGeneration"]
    lm_cfg["model_type"] = "gemma4"

    # Preserve cross-modal tokens the multimodal forward needs.
    for k in (
        "image_token_id", "boi_token_id", "eoi_token_id",
        "vision_soft_tokens_per_image",
        "audio_token_id", "boa_token_id", "eoa_token_id",
        "eoa_token_index", "video_token_id",
    ):
        if k in bf16_cfg and k not in lm_cfg:
            lm_cfg[k] = bf16_cfg[k]

    (output_dir / "config.json").write_text(json.dumps(lm_cfg, indent=2))
    log.info("Wrote merged config.json")

    # --- 6. Copy supporting files (tokenizer, processor_config, etc.) ---
    # Canonical sidecar set from common/model_io.py; ``tokenizer.model``
    # / ``vocab.json`` / ``merges.txt`` are kept for BPE-tokenizer
    # checkpoints — they're silently no-ops on Gemma 4 which ships only
    # ``tokenizer.json`` (fast tokenizer).
    from src.common.model_io import PROCESSOR_SIDECAR_FILES

    for fname in (*PROCESSOR_SIDECAR_FILES, "vocab.json", "merges.txt"):
        src = bf16_dir / fname  # bf16 dir is the authoritative source
        if not src.exists():
            src = lm_dir / fname
        if src.exists():
            shutil.copy2(src, output_dir / fname)

    # --- 7. Drop a splice manifest for provenance ---
    manifest = {
        "splice_version": 1,
        "lm_source": str(lm_dir),
        "bf16_source": str(bf16_dir),
        "include_audio": include_audio,
        "n_lm_tensors": len(lm_keys),
        "n_vision_tensors_added": len(bf16_keep),
        "total_bytes": out_size,
        "total_gb": out_size / 1e9,
    }
    (output_dir / "splice_manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote splice_manifest.json")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lm-dir", type=Path, required=True,
                        help="LM-only quantized output directory.")
    parser.add_argument("--bf16-dir", type=Path, required=True,
                        help="Original merged BF16 model directory.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write the spliced multimodal model.")
    parser.add_argument("--include-audio", action="store_true",
                        help="Bring in audio_tower + embed_audio (BF16). "
                             "Default: strip them (iOS doesn't use audio).")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args.lm_dir = args.lm_dir.expanduser().resolve()
    args.bf16_dir = args.bf16_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    manifest = splice(args.lm_dir, args.bf16_dir, args.output_dir,
                      include_audio=args.include_audio)
    log.info("DONE. Spliced model at %s (%.3f GB)",
             args.output_dir, manifest["total_gb"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
