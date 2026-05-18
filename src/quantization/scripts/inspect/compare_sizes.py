#!/usr/bin/env python3
"""Side-by-side per-submodule size diff between two quantized models.

Use this to understand why two methods produce different total sizes.
Runs on any host — only reads safetensors headers.

Usage:
    python -m scripts.inspect.compare_sizes <dir_a> <dir_b>
    python -m scripts.inspect.compare_sizes \\
        ./mlx_community_baseline ./unsloth_ud_9ee11f5 \\
        --label_a mlx-community --label_b unsloth-UD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.sizing import diff_directories, measure_directory  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir_a", type=Path)
    parser.add_argument("dir_b", type=Path)
    parser.add_argument("--label_a", default="A")
    parser.add_argument("--label_b", default="B")
    args = parser.parse_args(argv)

    a = measure_directory(args.dir_a)
    b = measure_directory(args.dir_b)
    print(f"{args.label_a}: {args.dir_a}")
    print(f"{args.label_b}: {args.dir_b}")
    print()
    print(diff_directories(a, b, label_a=args.label_a, label_b=args.label_b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
