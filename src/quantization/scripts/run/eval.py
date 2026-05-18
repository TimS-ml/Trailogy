#!/usr/bin/env python3
"""Run the eval sweep for one quantization variant.

Usage:
    # bf16 baseline (full PlantNet val + WikiText PPL + VQAv2):
    python -m scripts.run.eval \\
        --variant bf16_baseline \\
        --loader hf_bf16 \\
        --model_dir exports/sft-50k-fullproj/merged-bf16 \\
        --plantnet_val_jsonl finetune/data/val.jsonl \\
        --benchmarks plantnet_val wikitext_ppl vqav2_devtest \\
        --output_dir quantization/results/bf16_baseline

    # mlx_vlm INT4 (skip PPL because mlx_vlm.generate doesn't expose logprobs):
    python -m scripts.run.eval \\
        --variant mlx_vlm_g64 \\
        --loader mlx_vlm \\
        --model_dir results/mlx_vlm_g64 \\
        --plantnet_val_jsonl finetune/data/val.jsonl \\
        --benchmarks plantnet_val vqav2_devtest \\
        --output_dir quantization/results/mlx_vlm_g64

Subset eval for smoke tests:
    --plantnet_n 100 --vqav2_n 200 --wikitext_n 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# parents[2] = quantization/ (where src/ lives). Prior to the
# 0571c36 scripts/ reorg this was parents[1] = quantization/; after
# the reorg the eval module moved to scripts/run/, so we need one
# more level up. Adding parents[1] (= quantization/scripts/) also
# triggers a stdlib shadow because scripts/inspect/ shadows the
# stdlib `inspect` module used by dataclasses.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.sizing import measure_directory  # noqa: E402
from src.eval.model_loaders import (  # noqa: E402
    GPTQ_BACKEND_CHOICES,
    LOADER_REGISTRY,
)
from src.eval.runner import BENCHMARK_REGISTRY, RunnerConfig, run_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variant", required=True, help="Name for the eval output.")
    parser.add_argument(
        "--loader", required=True, choices=sorted(LOADER_REGISTRY),
    )
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["plantnet_val"],
        choices=sorted(BENCHMARK_REGISTRY),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--base_model_for_processor",
        default="unsloth/gemma-4-E2B-it",
        help="Used by the hf_bf16 loader if the merged dir lacks processor configs.",
    )
    parser.add_argument(
        "--gptq_backend",
        default="triton",
        choices=GPTQ_BACKEND_CHOICES,
        help=(
            "Kernel backend for the hf_gptq loader. Defaults to triton "
            "(compatible with every checkpoint; slowest at batch=1 decode). "
            "Use 'marlin' for w4g128 sym desc_act=False to get ~2-3x decode "
            "speedup; use 'exllama_v2' for desc_act=True checkpoints. "
            "Ignored when --loader is not hf_gptq."
        ),
    )

    # Per-benchmark knobs
    parser.add_argument("--plantnet_val_jsonl", type=Path)
    parser.add_argument("--plantnet_n", type=int, default=None)
    parser.add_argument("--vqav2_n", type=int, default=1000)
    parser.add_argument("--wikitext_n", type=int, default=200)

    # v4 conditional-FT camera-state gate. When the checkpoint was
    # trained with ``data.prompt_prefixes`` (camera-state markers
    # prepended to user prompts), eval MUST inject the same markers
    # or species_match collapses to ~0 (the model has only ever seen
    # ``[camera=on] What plant...`` and gets confused by raw text).
    # Set both flags for the production v4 contract; leave unset for
    # pre-v4 checkpoints. Routes through to PlantNetConfig.prompt_prefixes.
    parser.add_argument(
        "--prompt_prefix_camera_on",
        default=None,
        help=(
            "Literal string prepended to the first user turn for "
            "image-bearing PlantNet records (v4 camera-state gate). "
            "Typical value: '[camera=on] '. Leave unset for pre-v4 "
            "checkpoints. Mirrors data.prompt_prefixes.camera_on in "
            "the training YAML."
        ),
    )
    parser.add_argument(
        "--prompt_prefix_camera_off",
        default=None,
        help=(
            "Literal string prepended to the first user turn for "
            "text-only records. Typical value: '[camera=off] '. "
            "Mirrors data.prompt_prefixes.camera_off."
        ),
    )

    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--eval_seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if "plantnet_val" in args.benchmarks and args.plantnet_val_jsonl is None:
        parser.error("--plantnet_val_jsonl is required when plantnet_val is in --benchmarks")

    # Load the model via the selected loader.
    loader = LOADER_REGISTRY[args.loader]
    loader_kwargs: dict = {}
    if args.loader == "hf_bf16":
        loader_kwargs["device_map"] = args.device_map
        loader_kwargs["base_model_for_processor"] = args.base_model_for_processor
    elif args.loader in ("hf_gptq", "hf_gptq_hybrid"):
        loader_kwargs["base_model_for_processor"] = args.base_model_for_processor
        loader_kwargs["backend"] = args.gptq_backend
    handle = loader(args.model_dir, **loader_kwargs)

    # Resolve the v4 prompt_prefixes dict from CLI flags. Either flag
    # being set activates the dict; absent keys fall through to "no
    # prefix for that branch" inside build_vision_messages.
    prompt_prefixes: dict[str, str] | None = None
    if args.prompt_prefix_camera_on is not None or args.prompt_prefix_camera_off is not None:
        prompt_prefixes = {}
        if args.prompt_prefix_camera_on is not None:
            prompt_prefixes["camera_on"] = args.prompt_prefix_camera_on
        if args.prompt_prefix_camera_off is not None:
            prompt_prefixes["camera_off"] = args.prompt_prefix_camera_off

    # Build benchmark configs
    bench_cfgs: dict[str, dict] = {}
    if "plantnet_val" in args.benchmarks:
        bench_cfgs["plantnet_val"] = {
            "val_jsonl": str(args.plantnet_val_jsonl),
            "n_samples": args.plantnet_n,
        }
        if prompt_prefixes is not None:
            bench_cfgs["plantnet_val"]["prompt_prefixes"] = prompt_prefixes
    if "wikitext_ppl" in args.benchmarks:
        bench_cfgs["wikitext_ppl"] = {"n_segments": args.wikitext_n}
    if "vqav2_devtest" in args.benchmarks:
        bench_cfgs["vqav2_devtest"] = {"n_samples": args.vqav2_n}

    cfg = RunnerConfig(
        variant=args.variant,
        benchmarks=args.benchmarks,
        benchmark_configs=bench_cfgs,
        eval_seed=args.eval_seed,
        output_dir=args.output_dir,
    )
    payload = run_all(handle, cfg)

    # Attach on-disk size + which kernel produced the numbers. The
    # kernel only meaningfully varies for the GPTQ path; including it
    # for every variant keeps the JSON schema uniform across loaders.
    try:
        size = measure_directory(handle.model_dir)
        payload["model_size_gb"] = round(size.total_disk_bytes / (1 << 30), 3)
        payload["model_size_per_submodule_bytes"] = size.per_submodule_bytes
    except Exception as e:  # noqa: BLE001
        print(f"  (warning: size measurement failed: {e})")
    payload["loader"] = args.loader
    if args.loader in ("hf_gptq", "hf_gptq_hybrid"):
        payload["gptq_backend"] = args.gptq_backend
    # Re-write the JSON with size + kernel info attached.
    (args.output_dir / "eval.json").write_text(__import__("json").dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
