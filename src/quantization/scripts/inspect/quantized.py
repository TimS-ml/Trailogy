#!/usr/bin/env python3
"""Print a size + dtype report for a quantized model directory.

Runs on any host (no CUDA, no MLX) — only reads safetensors headers.

Usage:
    python -m scripts.inspect.quantized <model_dir>
    python -m scripts.inspect.quantized <model_dir> --json out.json

The primary tool for understanding why one method produces 3.58 GB
and another produces 4.52 GB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``quantization`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.sizing import measure_directory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--json",
        type=Path,
        help="Also write structured size report as JSON to this path.",
    )
    args = parser.parse_args(argv)

    if not args.model_dir.is_dir():
        print(f"Not a directory: {args.model_dir}", file=sys.stderr)
        return 2

    report = measure_directory(args.model_dir)
    print(report.format_report())

    if args.json:
        payload = {
            "directory": str(report.directory),
            "total_disk_bytes": report.total_disk_bytes,
            "total_safetensors_bytes": report.total_safetensors_bytes,
            "non_safetensors_bytes": report.non_safetensors_bytes,
            "per_submodule_bytes": report.per_submodule_bytes,
            "per_submodule_avg_bits": report.per_submodule_avg_bits,
            "dtype_histogram": {
                dt: list(vals) for dt, vals in report.dtype_histogram.items()
            },
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
