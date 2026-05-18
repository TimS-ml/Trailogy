#!/usr/bin/env python3
"""Bake an EoRA adapter into an MLX g-affine quantized checkpoint.

The previous (broken) merge went through ``mlx_vlm.load`` → in-memory
model → save_state, which silently rewrote ALL tensors (vision_tower,
audio_tower, layernorms) at bf16 round-trip precision and a fresh
non-deterministic random init for some norms. The resulting model
produced pad-spam.

This script does the merge at the **safetensors dict level**: only the
315 quantized linear layers covered by the EoRA adapter get their
``{weight, scales, biases}`` triplet rewritten. Every other tensor
(vision_tower.*, audio_tower.*, embed_*.*, norm.weight, etc.) is
copied through byte-for-byte.

Math per covered layer:
    W_bf16   = mx.dequantize(weight, scales, biases, g, bits)
    Δ        = lora_b.T @ lora_a.T            # (out, in)
    W_new    = W_bf16 + Δ
    weight', scales', biases' = mx.quantize(W_new.astype(bf16), g, bits)

The adapter file keys are like ``model.layers.0.mlp.down_proj.lora_a``
(no ``language_model.`` prefix). The quant safetensors keys are like
``language_model.model.layers.0.mlp.down_proj.weight``. The mapping is
just prefix-stripping.

Usage:
    python -m scripts.run.merge_eora_into_mlx \\
        --quant-dir   results/<sft>/mlx_g64 \\
        --adapter     results/<sft>/eora/adapters_r64.safetensors \\
        --rank        64 \\
        --output-dir  results/<sft>/mlx_g64_eora64-merge
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

log = logging.getLogger("merge_eora_into_mlx")


def main(argv: list[str] | None = None) -> int:
    import mlx.core as mx

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quant-dir", type=Path, required=True,
                        help="Source MLX quantized dir (must have format=mlx safetensors).")
    parser.add_argument("--adapter", type=Path, required=True,
                        help="EoRA adapter safetensors (save_adapters output).")
    parser.add_argument("--rank", type=int, default=None,
                        help="Truncate adapter to this rank (default: use saved rank).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lang-prefix", default="language_model.",
                        help="Prefix to strip from quant keys to match adapter keys.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    src_st = args.quant_dir / "model.safetensors"
    assert src_st.exists(), f"missing {src_st}"

    log.info("Reading quant dict: %s", src_st)
    weights = mx.load(str(src_st))
    log.info("  %d tensors loaded", len(weights))

    # Read quantization params from config.json (group_size, bits)
    with open(args.quant_dir / "config.json") as f:
        cfg = json.load(f)
    q = cfg["quantization"]
    group_size = int(q["group_size"])
    bits = int(q["bits"])
    log.info("  quant: group_size=%d, bits=%d, mode=%s",
             group_size, bits, q.get("mode"))

    log.info("Reading adapter: %s", args.adapter)
    raw_adp = mx.load(str(args.adapter))
    base_keys = sorted({
        k[: -len(".lora_a")] for k in raw_adp if k.endswith(".lora_a")
    })
    log.info("  %d layer adapter pairs", len(base_keys))

    # Build mapping: adapter key (`model.layers.0.mlp.down_proj`)
    # → safetensors weight key (`language_model.model.layers.0.mlp.down_proj.weight`)
    n_merged = 0
    n_missing = 0
    n_truncated = 0
    total_l2 = 0.0
    max_abs = 0.0

    t0 = time.perf_counter()
    for bk in base_keys:
        # Map to safetensors prefix
        st_prefix = f"{args.lang_prefix}{bk}"
        wk = f"{st_prefix}.weight"
        sk = f"{st_prefix}.scales"
        biask = f"{st_prefix}.biases"
        if wk not in weights or sk not in weights or biask not in weights:
            log.warning("Skip %s: not found in quant dict (looked for %s)", bk, wk)
            n_missing += 1
            continue

        w_q = weights[wk]
        scales = weights[sk]
        biases = weights[biask]

        # Dequantize → bf16 then float32 for math
        W = mx.dequantize(w_q, scales=scales, biases=biases,
                          group_size=group_size, bits=bits).astype(mx.float32)
        out_dim, in_dim = W.shape

        lora_a = raw_adp[f"{bk}.lora_a"]  # (in_dim, R_saved)
        lora_b = raw_adp[f"{bk}.lora_b"]  # (R_saved, out_dim)
        r_saved = int(lora_a.shape[1])
        if args.rank is not None and args.rank < r_saved:
            lora_a = lora_a[:, -args.rank:]
            lora_b = lora_b[-args.rank:, :]
            n_truncated += 1
        # Build correction shape (out, in) = lora_b^T @ lora_a^T
        # = (out_dim, r) @ (r, in_dim) — but lora_b is (r, out) so lora_b.T is (out, r),
        # lora_a is (in, r) so lora_a.T is (r, in). product = (out, in). ✓
        corr = (lora_b.T.astype(mx.float32) @ lora_a.T.astype(mx.float32))
        # Sanity: shapes must match
        assert corr.shape == W.shape, (
            f"{bk}: corr shape {corr.shape} != W shape {W.shape}"
        )
        W_new = W + corr

        total_l2 += float(mx.sum(corr * corr).item()) ** 0.5
        max_abs = max(max_abs, float(mx.max(mx.abs(corr)).item()))

        # Re-quantize back to int4 packed (cast to bf16 first as in MLX convention)
        w_q_new, scales_new, biases_new = mx.quantize(
            W_new.astype(mx.bfloat16), group_size=group_size, bits=bits
        )
        weights[wk] = w_q_new
        weights[sk] = scales_new
        weights[biask] = biases_new
        n_merged += 1

    elapsed = time.perf_counter() - t0
    log.info(
        "Merged %d/%d layers in %.1fs  (truncated=%d, missing=%d)  "
        "avg L2/layer=%.4f  max|Δ|=%.4f",
        n_merged, len(base_keys), elapsed, n_truncated, n_missing,
        total_l2 / max(n_merged, 1), max_abs,
    )

    # Write merged safetensors with format=mlx metadata (required by
    # mlx_vlm.load to skip its add-model-prefix sanitize)
    out_st = args.output_dir / "model.safetensors"
    log.info("Saving: %s", out_st)
    mx.save_safetensors(str(out_st), weights, metadata={"format": "mlx"})

    # Copy companion files (config, tokenizer, etc.) byte-for-byte
    for name in [
        "config.json", "generation_config.json", "processor_config.json",
        "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "README.md",
    ]:
        src = args.quant_dir / name
        if src.exists():
            shutil.copy2(src, args.output_dir / name)

    log.info("Done. Output: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
