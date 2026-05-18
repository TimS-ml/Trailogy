#!/usr/bin/env python3
"""CLI for the GPTQ + torchao hybrid post-processor.

Takes an existing GPTQModel output directory (e.g.
``results/gptq_w4g128_da1/``) and produces a hybrid artifact with
audio stripped and the two ``nn.Embedding`` tables quantized to
int4-packed uint8. See ``docs/quantization/B1-torchao-vs-gptqmodel.md``
for the design rationale and ``src.methods.gptq_torchao_hybrid`` for
the implementation.

Usage
-----
::

    python -m scripts.run.gptq_torchao_hybrid \\
        --input  quantization/results/gptq_w4g128_da1 \\
        --output quantization/results/gptq_w4g128_da1_hybrid \\
        --embed-per-layer-bits 4 --embed-per-layer-group-size 128 \\
        --embed-tokens-bits 4    --embed-tokens-group-size    128 \\
        --strip-audio

Set ``--embed-per-layer-bits none`` to keep the per-layer table at
bf16 (ablation variant). Set ``--no-strip-audio`` to keep the audio
tower (for cross-checking that audio strip is loss-free, since the
text/vision paths don't touch it).

The output dir is a self-contained HF-format checkpoint plus a
``config.json`` ``hybrid_quant`` block. Loading at eval time:

.. code-block:: python

    from transformers import AutoModelForImageTextToText
    from src.methods.gptq_torchao_hybrid import load_hybrid_embeddings

    model = AutoModelForImageTextToText.from_pretrained(
        hybrid_dir, torch_dtype="bfloat16", device_map="cuda:0",
    )  # will warn about missing embed_tokens(.weight) etc — expected
    load_hybrid_embeddings(model, hybrid_dir, device="cuda:0")
    # model is now ready for inference
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ``src.methods.*`` resolves via the CWD that ``python -m
# scripts.run.gptq_torchao_hybrid`` automatically adds to sys.path
# (the ``quantization`` directory). Do NOT do
# ``sys.path.insert(0, parents[1])`` — that would add
# ``quantization/scripts/`` to sys.path[0], which then shadows the
# stdlib ``inspect`` module with the local ``scripts/inspect/``
# subpackage. The shadow only surfaces once something downstream
# tries to import ``inspect`` (e.g. ``import torch`` →
# ``typing_extensions`` → ``inspect.signature``), and that surfaces
# as a confusing ``AttributeError: module 'inspect' has no attribute
# 'signature'``. Other run scripts in this directory use the
# ``parents[1]`` pattern; they get away with it only because they
# import torch lazily and the shadow doesn't fire during ``--help``.

log = logging.getLogger("run_gptq_torchao_hybrid")


def _parse_bits(value: str) -> int | None:
    """Accept 'none' / '4' / '8' etc. ``none`` ⇒ keep bf16."""
    if value.lower() == "none":
        return None
    iv = int(value)
    if iv != 4:
        raise argparse.ArgumentTypeError(
            f"only --bits=4 is implemented today; got {iv}"
        )
    return iv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Existing GPTQ output directory. Required unless --smoke.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Target directory for the hybrid artifact (will be created). "
             "Required unless --smoke.",
    )
    parser.add_argument(
        "--strip-audio", dest="strip_audio", action="store_true", default=True,
        help="Drop audio_tower + embed_audio tensors (default: on).",
    )
    parser.add_argument(
        "--no-strip-audio", dest="strip_audio", action="store_false",
        help="Keep audio_tower + embed_audio (ablation only).",
    )
    parser.add_argument(
        "--embed-per-layer-bits", type=_parse_bits, default=4,
        help="Quant bits for embed_tokens_per_layer (4 or 'none'). Default 4.",
    )
    parser.add_argument(
        "--embed-per-layer-group-size", type=int, default=128,
    )
    parser.add_argument(
        "--embed-per-layer-mapping", default="asymmetric",
        choices=("symmetric", "asymmetric"),
    )
    parser.add_argument(
        "--embed-tokens-bits", type=_parse_bits, default=4,
        help="Quant bits for embed_tokens (4 or 'none'). Default 4.",
    )
    parser.add_argument(
        "--embed-tokens-group-size", type=int, default=128,
    )
    parser.add_argument(
        "--embed-tokens-mapping", default="asymmetric",
        choices=("symmetric", "asymmetric"),
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for the torchao quantization pass (default cuda).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Don't run the full pipeline; just verify deps.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    from src.methods.gptq_torchao_hybrid import (
        HybridConfig,
        quantize_hybrid,
        smoke_check,
    )

    if args.smoke:
        result = smoke_check()
        for k, v in result.items():
            print(f"{k}: {v}")
        return 0 if result["deps_available"] else 1

    if args.input is None or args.output is None:
        log.error("--input and --output are required (omit only with --smoke).")
        return 2
    if not args.input.is_dir():
        log.error("--input does not exist: %s", args.input)
        return 2

    config = HybridConfig(
        strip_audio=args.strip_audio,
        embed_per_layer_bits=args.embed_per_layer_bits,
        embed_per_layer_group_size=args.embed_per_layer_group_size,
        embed_per_layer_mapping=args.embed_per_layer_mapping,
        embed_tokens_bits=args.embed_tokens_bits,
        embed_tokens_group_size=args.embed_tokens_group_size,
        embed_tokens_mapping=args.embed_tokens_mapping,
        device=args.device,
    )
    quantize_hybrid(args.input, args.output, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
