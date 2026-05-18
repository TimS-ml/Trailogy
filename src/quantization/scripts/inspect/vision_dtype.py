#!/usr/bin/env python3
"""Tripwire: assert the vision tower stayed bf16 (or fp32) in a quantized dir.

Reads safetensors headers only — no CUDA, no MLX. Designed as a CI hook
between any quant run and an iOS / HF upload. Exits 0 if the invariant
holds, 1 otherwise.

Background: see ``07-quantization/docs/06-bnb-nf4-vision-collapse.md``.
Empirically, NF4-ing the SigLIP vision tower drops PlantNet
species_match from 70.6 % to 0.1 %. The deployable variants (GPTQ
w4g128, MLX VLM w4g64) all leave ``vision_tower.*`` at bf16. This
script enforces that as a build-time invariant.

Usage:
    python -m scripts.inspect.vision_dtype <model_dir>
    python -m scripts.inspect.vision_dtype <model_dir> \\
        --min_bytes 250000000 --max_bytes 400000000

Defaults reflect Gemma 4 E2B's SigLIP tower (~319 MB at bf16).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``quantization`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.safetensors_io import enumerate_directory  # noqa: E402


# Acceptable dtypes for any tensor under ``vision_tower.*``.
# - BF16 / F32: the only dtypes we want to ship for inference accuracy.
# - F16: borderline acceptable (MLX VLM sometimes emits FP16 scales for
#   other submodules; we don't expect it inside vision_tower at bf16,
#   but it is not a *quantized* dtype, so we keep it permissive).
ALLOWED_VISION_DTYPES = frozenset({"BF16", "F32", "F16"})

# Sanity bounds on the vision tower's total on-disk size. Gemma 4 E2B's
# SigLIP tower is ~319 MB at bf16; allow a generous ±25 % window so we
# catch both "tower was stripped to zero stubs" and "tower somehow grew
# 2x". These can be overridden via CLI flags for other base models.
DEFAULT_MIN_BYTES = 250_000_000  # 0.25 GB
DEFAULT_MAX_BYTES = 400_000_000  # 0.40 GB


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--min_bytes",
        type=int,
        default=DEFAULT_MIN_BYTES,
        help="Minimum acceptable total bytes under vision_tower.* "
        f"(default: {DEFAULT_MIN_BYTES:,}).",
    )
    parser.add_argument(
        "--max_bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum acceptable total bytes under vision_tower.* "
        f"(default: {DEFAULT_MAX_BYTES:,}).",
    )
    args = parser.parse_args(argv)

    if not args.model_dir.is_dir():
        print(f"Not a directory: {args.model_dir}", file=sys.stderr)
        return 2

    entries = enumerate_directory(args.model_dir)
    vt_entries = [t for t in entries if "vision_tower" in t.name]

    if not vt_entries:
        print(
            f"FAIL: no vision_tower.* tensors found in {args.model_dir}. "
            "Did mlx_lm strip the vision tower?",
            file=sys.stderr,
        )
        return 1

    bad = [t for t in vt_entries if t.dtype not in ALLOWED_VISION_DTYPES]
    total_bytes = sum(t.nbytes for t in vt_entries)

    if bad:
        print(
            f"FAIL: {len(bad)} vision_tower.* tensor(s) in disallowed "
            f"dtype(s). vision_tower MUST be bf16/fp32 for accurate "
            f"inference. Examples:",
            file=sys.stderr,
        )
        for t in bad[:8]:
            print(f"  {t.dtype:<6}  {t.name}", file=sys.stderr)
        if len(bad) > 8:
            print(f"  ... ({len(bad) - 8} more)", file=sys.stderr)
        return 1

    if total_bytes < args.min_bytes:
        print(
            f"FAIL: vision_tower total bytes {total_bytes:,} < min "
            f"{args.min_bytes:,}. Tower may have been stripped to stubs.",
            file=sys.stderr,
        )
        return 1
    if total_bytes > args.max_bytes:
        print(
            f"FAIL: vision_tower total bytes {total_bytes:,} > max "
            f"{args.max_bytes:,}. Unexpected — re-check the export.",
            file=sys.stderr,
        )
        return 1

    print(
        f"OK: vision_tower in {args.model_dir} is bf16-clean. "
        f"{len(vt_entries)} tensors, {total_bytes / 1e6:.1f} MB on disk."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
