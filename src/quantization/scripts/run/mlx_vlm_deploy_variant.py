#!/usr/bin/env python3
"""Run a single mlx_vlm.convert deploy-quant variant, with optional
extension to the default skip list.

Background
----------
``mlx_vlm.convert`` exposes ``--q-bits``, ``--q-group-size``,
``--q-mode``, and ``--quant-predicate`` (named mixed-precision recipe)
as CLI flags. The set of modules that get quantized is controlled by
``mlx_vlm.utils.skip_multimodal_module``, which skips ``vision_tower.*``
and ``audio_tower.*`` by default — but NOT ``embed_vision.*`` (the
multimodal projector) or ``embed_audio.*``.

For Gemma 4 SFTs that trained the projector at full precision
(``tune_projector: true``), keeping ``embed_vision`` bf16 at deploy
time is a candidate fix for the iOS-INT4 accuracy collapse observed in
baseline deploy-quant sweeps.

This script wraps ``mlx_vlm.convert.convert()`` in-process so we can
supply a custom ``quant_predicate`` callable that adds further skip
substrings beyond the default. CLI knobs (``--q-bits`` / ``--q-group-size``
/ ``--q-mode`` / ``--quant-predicate`` recipe-name) are passed through
unchanged. The recipe-name case (``mixed_*``) and the custom-skip case
are mutually exclusive: a recipe name installs its own predicate
internally, and we don't try to wrap it.

Usage
-----
    # Variant 1 — skip embed_vision (the SFT'd projector):
    source quantization/scripts/_env/_mlx_env.sh
    $MLX_PYTHON -m scripts.run.mlx_vlm_deploy_variant \
        --src   quantization/results/baseline2_qlora_safemerged_bf16 \
        --dst   quantization/results/baseline2_qlora_mlx_vlm_g64_skip_ev \
        --q-bits 4 --q-group-size 64 --q-mode affine \
        --also-skip embed_vision

    # Variant 2 — group_size 32, no extra skip:
    $MLX_PYTHON -m scripts.run.mlx_vlm_deploy_variant \
        --src ... --dst ... \
        --q-bits 4 --q-group-size 32 --q-mode affine

    # Variant 3/4 — mixed-precision recipe (no extra skip supported):
    $MLX_PYTHON -m scripts.run.mlx_vlm_deploy_variant \
        --src ... --dst ... \
        --quant-predicate-recipe mixed_4_8

The script wall-time is dominated by ``save_weights`` (~2 min for the
full 2,649-tensor multimodal tree); custom-predicate computation itself
is microseconds.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

# Lazy mlx_vlm imports — heavy and platform-sensitive; deferred until
# after argparse so --help works without the env being set up.

log = logging.getLogger("run_mlx_vlm_deploy_variant")


def _build_extended_skip_predicate(
    extra_skips: list[str],
) -> Callable[[str, object], bool]:
    """Return a predicate that mirrors mlx_vlm's default skip behavior
    AND additionally returns False for any path containing one of the
    ``extra_skips`` substrings.

    Substring match (not anchored prefix) — same convention as
    ``skip_multimodal_module``. Each extra_skip is matched as a plain
    substring of the dotted module path.
    """
    from mlx_vlm.utils import skip_multimodal_module

    def predicate(path: str, module: object) -> bool:
        if skip_multimodal_module(path):
            return False
        for needle in extra_skips:
            if needle in path:
                return False
        # ``hasattr(module, "to_quantized")`` mirrors what the upstream
        # ``mixed_quant_predicate`` does: returning True for a module
        # that has no ``to_quantized`` is a no-op (quantize_model
        # skips it anyway), but being explicit avoids surprises on
        # custom Linear subclasses (e.g. ScaledLinear in Gemma 4).
        if not hasattr(module, "to_quantized"):
            return False
        return True

    return predicate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--src", type=Path, required=True, help="bf16 HF dir")
    parser.add_argument("--dst", type=Path, required=True, help="output mlx dir")
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--q-mode",
        default="affine",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
    )
    parser.add_argument(
        "--also-skip",
        nargs="+",
        default=[],
        help=(
            "Substrings to additionally exclude from quantization (e.g. "
            "'embed_vision' to keep the SFT'd multimodal projector at bf16). "
            "Each entry is matched as a plain substring of the dotted module "
            "path. Mutually exclusive with --quant-predicate-recipe."
        ),
    )
    parser.add_argument(
        "--quant-predicate-recipe",
        default=None,
        help=(
            "Named mixed-precision recipe (e.g. mixed_4_8). When set, the "
            "recipe's own predicate is used and --also-skip is ignored "
            "(a recipe controls its own skip logic)."
        ),
    )
    parser.add_argument(
        "--dtype",
        default=None,
        help="Cast all floating params to this dtype before quant (e.g. bfloat16).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved arguments and exit without invoking convert.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.src.is_dir():
        log.error("--src does not exist: %s", args.src)
        return 2

    args.dst.mkdir(parents=True, exist_ok=True)

    if args.also_skip and args.quant_predicate_recipe:
        log.error(
            "--also-skip and --quant-predicate-recipe are mutually exclusive "
            "(a recipe installs its own predicate)."
        )
        return 2

    log.info("src              = %s", args.src)
    log.info("dst              = %s", args.dst)
    log.info("q_bits           = %d", args.q_bits)
    log.info("q_group_size     = %d", args.q_group_size)
    log.info("q_mode           = %s", args.q_mode)
    log.info("also_skip        = %s", args.also_skip)
    log.info("predicate_recipe = %s", args.quant_predicate_recipe)
    log.info("dtype            = %s", args.dtype)

    if args.dry_run:
        log.info("--dry-run: exiting without invoking convert.")
        return 0

    # Build the predicate (callable or recipe-name string), then call
    # ``mlx_vlm.convert.convert``. The string-vs-callable branching is
    # handled inside convert().
    from mlx_vlm.convert import convert

    if args.quant_predicate_recipe:
        quant_predicate = args.quant_predicate_recipe
    elif args.also_skip:
        quant_predicate = _build_extended_skip_predicate(list(args.also_skip))
    else:
        quant_predicate = None  # convert() falls back to its default

    convert(
        hf_path=str(args.src),
        mlx_path=str(args.dst),
        quantize=True,
        q_group_size=args.q_group_size,
        q_bits=args.q_bits,
        q_mode=args.q_mode,
        dtype=args.dtype,
        quant_predicate=quant_predicate,
    )
    log.info("Done. Output → %s", args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
