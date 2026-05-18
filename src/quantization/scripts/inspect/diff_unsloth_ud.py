#!/usr/bin/env python3
"""Diff the Unsloth UD MLX 4-bit HF releases to extract the promotion list.

Why: `unsloth/gemma-4-E2B-it-UD-MLX-4bit` is 3.55 GB at commit `9ee11f5`
and 4.52 GB at HEAD. Most likely cause: HEAD "promotes" more tensors
from INT4 → fp16/bf16. Identifying exactly which tensors got promoted
gives us a recipe we can replicate in
``quantization/methods/unsloth_ud.py`` to ship under the 4 GB ceiling.

This script:
1. Downloads (or reuses cache) both the OLD (3.55 GB) and HEAD (4.52 GB)
   safetensors via huggingface_hub.
2. Optionally compares against ``mlx-community/gemma-4-e2b-it-4bit``
   (3.58 GB) for a 3-way diff.
3. Prints a per-tensor table: which tensors differ in dtype/size
   between revisions.
4. Outputs a YAML-compatible promotion_keys list ready to paste into
   ``quantization/configs/unsloth_ud_9ee11f5.yaml``.

Runs anywhere (no CUDA, no MLX). Only needs network + huggingface_hub.

Usage:
    python -m scripts.inspect.diff_unsloth_ud
    python -m scripts.inspect.diff_unsloth_ud --three_way
    python -m scripts.inspect.diff_unsloth_ud --output_yaml out.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.safetensors_io import enumerate_directory  # noqa: E402
from src.common.sizing import fmt_bytes  # noqa: E402

log = logging.getLogger("diff_unsloth_ud")


UNSLOTH_REPO = "unsloth/gemma-4-E2B-it-UD-MLX-4bit"
UNSLOTH_OLD_COMMIT = "9ee11f5"  # 3.55 GB recipe
MLX_COMMUNITY_REPO = "mlx-community/gemma-4-e2b-it-4bit"


def _download(repo_id: str, revision: str | None = None) -> Path:
    """Snapshot-download a checkpoint, reusing the HF cache if present.

    Note: these checkpoints are 3.5–4.5 GB each and may take 5-15 min
    on a residential connection. The download is **resumable** — if
    interrupted, re-run and HF cache will continue where it stopped.
    """
    from huggingface_hub import snapshot_download

    log.info(
        "Fetching %s @ %s (resumable; reuses HF cache)",
        repo_id, revision or "main",
    )
    # Note: ``resume_download`` was deprecated in recent huggingface_hub —
    # downloads resume automatically now.
    p = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        allow_patterns=["*.safetensors", "*.json"],
    )
    return Path(p)


def _by_name(directory: Path) -> dict[str, "TensorEntry"]:
    return {e.name: e for e in enumerate_directory(directory)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--three_way",
        action="store_true",
        help="Also include mlx-community/gemma-4-e2b-it-4bit (3.58 GB) in the diff.",
    )
    parser.add_argument(
        "--output_yaml",
        type=Path,
        help="If set, write a YAML-formatted promotion_keys list to this file.",
    )
    parser.add_argument(
        "--bits_threshold",
        type=float,
        default=8.0,
        help="A tensor is considered 'promoted' if its bits/elt is above this value. "
        "INT4 packed weights are typically 4-6 bits/elt; fp16/bf16 are 16; "
        "8 is a safe demarcation.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    old_dir = _download(UNSLOTH_REPO, revision=UNSLOTH_OLD_COMMIT)
    new_dir = _download(UNSLOTH_REPO)  # HEAD
    old = _by_name(old_dir)
    new = _by_name(new_dir)

    if args.three_way:
        mlxc_dir = _download(MLX_COMMUNITY_REPO)
        mlxc = _by_name(mlxc_dir)
    else:
        mlxc = None

    # 1. Tensors present in both old and new, with different dtype OR
    #    different on-disk size.
    differing: list[tuple[str, "TensorEntry", "TensorEntry"]] = []
    for name in sorted(set(old) & set(new)):
        e_old = old[name]
        e_new = new[name]
        if e_old.dtype != e_new.dtype or e_old.nbytes != e_new.nbytes:
            differing.append((name, e_old, e_new))

    log.info("Tensors differing between OLD (%s) and HEAD: %d",
             UNSLOTH_OLD_COMMIT, len(differing))

    # 2. Group by submodule for readability.
    by_submod: dict[str, list[tuple[str, "TensorEntry", "TensorEntry"]]] = defaultdict(list)
    for name, eo, en in differing:
        submod = _submod_of(name)
        by_submod[submod].append((name, eo, en))

    print()
    print(f"{'tensor':<70s}  {'old_dtype':>10s}  {'new_dtype':>10s}  {'old_bits':>9s}  {'new_bits':>9s}  {'delta_bytes':>14s}")
    print("-" * 132)
    for submod in sorted(by_submod):
        for name, eo, en in by_submod[submod]:
            print(
                f"{name:<70s}  "
                f"{eo.dtype:>10s}  {en.dtype:>10s}  "
                f"{eo.bits_per_element:>9.2f}  {en.bits_per_element:>9.2f}  "
                f"{fmt_bytes(en.nbytes - eo.nbytes):>14s}"
            )

    # 3. Promotion list (HEAD-side): tensors that are HIGH precision in
    #    HEAD compared to OLD. The HEAD is the larger 4.52 GB checkpoint,
    #    so HEAD = promotions of bits/elt vs OLD.
    promotion_keys: list[str] = []
    for name, eo, en in differing:
        if en.bits_per_element > args.bits_threshold > eo.bits_per_element:
            promotion_keys.append(name)

    print()
    print(f"Tensors PROMOTED in HEAD (above {args.bits_threshold} bits/elt): {len(promotion_keys)}")
    print("These are the tensors to keep at higher precision if you want")
    print("HEAD's recipe. To replicate the OLD 3.55 GB recipe, DO NOT include them.")
    print()

    # 4. The OLD recipe's promotion list (tensors that are HIGH precision
    #    in OLD too). That's what we actually want to replicate.
    old_high_precision = [
        name for name, e in old.items() if e.bits_per_element > args.bits_threshold
    ]
    print(f"Tensors at HIGH precision in OLD ({UNSLOTH_OLD_COMMIT}, 3.55 GB recipe): {len(old_high_precision)}")
    if old_high_precision:
        print("Sample (first 20):")
        for n in sorted(old_high_precision)[:20]:
            print(f"  - {n}  ({old[n].dtype}, {old[n].bits_per_element:.2f} bits/elt)")

    if mlxc is not None:
        print()
        print("Three-way comparison with mlx-community/gemma-4-e2b-it-4bit (3.58 GB):")
        mlxc_high = [n for n, e in mlxc.items() if e.bits_per_element > args.bits_threshold]
        print(f"  Tensors at HIGH precision in mlx-community: {len(mlxc_high)}")
        # Unique-to-old (vs mlx-community)
        only_in_old = set(old_high_precision) - set(mlxc_high)
        only_in_mlxc = set(mlxc_high) - set(old_high_precision)
        print(f"  Promoted in OLD-unsloth but NOT in mlx-community: {len(only_in_old)}")
        for n in sorted(only_in_old)[:10]:
            print(f"    - {n}")
        print(f"  Promoted in mlx-community but NOT in OLD-unsloth: {len(only_in_mlxc)}")
        for n in sorted(only_in_mlxc)[:10]:
            print(f"    - {n}")

    if args.output_yaml:
        yaml_keys = sorted(old_high_precision)
        lines = ["# Promotion keys extracted from unsloth UD OLD recipe (3.55 GB)."]
        lines.append(f"# Source: {UNSLOTH_REPO} @ {UNSLOTH_OLD_COMMIT}")
        lines.append(f"# Bits-per-elt threshold: {args.bits_threshold}")
        lines.append(f"# Total promoted tensors: {len(yaml_keys)}")
        lines.append("promotion_keys:")
        for k in yaml_keys:
            # Strip leading common prefixes to make the key list more
            # portable (works under different PEFT/HF wrapping).
            lines.append(f"  - {k!r}")
        args.output_yaml.write_text("\n".join(lines) + "\n")
        log.info("Wrote promotion_keys YAML to %s", args.output_yaml)

    return 0


def _submod_of(name: str) -> str:
    for tok in (
        "language_model.",
        "vision_tower.",
        "audio_tower.",
        "embed_vision.",
        "embed_audio.",
    ):
        if tok in name:
            return tok.rstrip(".")
    return "other"


if __name__ == "__main__":
    sys.exit(main())
