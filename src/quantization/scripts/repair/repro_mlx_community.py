#!/usr/bin/env python3
"""Reproduce `mlx-community/gemma-4-e2b-it-4bit` from the bf16 base.

Baseline reproduction gate. If this script produces an output directory
within ~5% of 3.58 GB and the per-submodule size table matches
mlx-community's safetensors header (modulo metadata), we have a verified
baseline pipeline.

Requires:
    - Apple Silicon Mac with mlx + mlx_vlm installed
    - `unsloth/gemma-4-E2B-it` bf16 cached locally or accessible via HF
    - Network to download `mlx-community/gemma-4-e2b-it-4bit` for comparison

Usage:
    python -m scripts.repair.repro_mlx_community \\
        --base_model unsloth/gemma-4-E2B-it \\
        --output_dir results/mlx_community_repro

To skip the comparison step (just produce the artifact):
    python -m scripts.repair.repro_mlx_community --skip_compare ...
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.sizing import diff_directories, measure_directory  # noqa: E402
from src.methods.mlx_vlm_baseline import (  # noqa: E402
    MLXBaselineConfig,
    quantize,
)

log = logging.getLogger("repro_mlx_community")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base_model", default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--reference_model",
        default="mlx-community/gemma-4-e2b-it-4bit",
        help="HF repo id of the reference quantized model to diff against.",
    )
    parser.add_argument(
        "--skip_compare",
        action="store_true",
        help="Skip downloading + diffing the reference model.",
    )
    parser.add_argument("--q_bits", type=int, default=4)
    parser.add_argument("--q_group_size", type=int, default=64)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # The base model needs to be on disk as a directory (or a HF cache path).
    # mlx_vlm.convert accepts an HF repo id directly. Let it resolve.
    log.info("Quantizing %s with q_bits=%d, group_size=%d",
             args.base_model, args.q_bits, args.q_group_size)
    cfg = MLXBaselineConfig(
        quantize=True, q_bits=args.q_bits, q_group_size=args.q_group_size
    )
    out_dir = quantize(Path(args.base_model), args.output_dir, cfg)
    our_size = measure_directory(out_dir)
    log.info("Our quantized output:")
    log.info(our_size.format_report())

    if args.skip_compare:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log.error("huggingface_hub not installed; --skip_compare to bypass.")
        return 2
    ref_dir = Path(
        snapshot_download(repo_id=args.reference_model, allow_patterns=["*.safetensors", "*.json"])
    )
    ref_size = measure_directory(ref_dir)
    log.info("Reference (%s):", args.reference_model)
    log.info(ref_size.format_report())

    log.info("Diff (reference vs ours):")
    log.info(
        diff_directories(ref_size, our_size, label_a="reference", label_b="ours")
    )

    # Soft pass criterion: within 5% on total bytes.
    delta = abs(our_size.total_disk_bytes - ref_size.total_disk_bytes)
    pct = delta / ref_size.total_disk_bytes * 100 if ref_size.total_disk_bytes else 100
    log.info("Total size delta: %.2f%%", pct)
    if pct > 5.0:
        log.warning(
            "Repro is >5%% off from reference. Investigate per-tensor "
            "dtypes before claiming the baseline is reproduced."
        )
        return 1
    log.info("OK — repro within 5%% of reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
