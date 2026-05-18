#!/usr/bin/env python3
"""One-shot quantization runner: merge LoRA → bf16 → quantize.

Routes to the per-method module based on ``--method``. The
``input``-side of every method is the same (a bf16 multimodal Gemma 4 E2B
optionally with a LoRA adapter to merge); the ``output``-side varies
per method.

Usage:
    python -m scripts.run.quant \\
        --method mlx_vlm_g64 \\
        --base_model unsloth/gemma-4-E2B-it \\
        --adapter outputs/.../final-adapter \\
        --output_dir results/mlx_vlm_g64

    # Skip merge if you already have a merged bf16 dir:
    python -m scripts.run.quant \\
        --method mlx_vlm_g64 \\
        --merged_dir exports/sft-merged \\
        --output_dir results/mlx_vlm_g64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("run_quant")


METHOD_REGISTRY = {
    "mlx_vlm_g32",
    "mlx_vlm_g64",
    "mlx_vlm_g128",
    "unsloth_ud",
    "gptq",
    "awq",
    "bnb_nf4",
    "qat_export",
}


def _filter_dataclass_kwargs(cls, raw: dict) -> dict:
    """Keep only fields that ``cls`` actually declares; drop the rest.

    Used to silently ignore YAML keys that don't map to a real
    dataclass field (e.g. comments / forward-compat fields) instead of
    raising ``TypeError`` deep in the dispatcher.
    """
    import dataclasses

    valid = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in raw.items() if k in valid}


def dispatch(
    method: str, merged_dir: Path, output_dir: Path, extra: dict | None = None
) -> Path:
    extra = extra or {}
    # YAML config is loaded by ``main`` and forwarded here so the
    # dispatcher stays usable from tests / programmatic callers without
    # a config file. Per-method overrides live under their own key.
    yaml_cfg: dict = extra.get("config_yaml") or {}

    if method in {"mlx_vlm_g32", "mlx_vlm_g64", "mlx_vlm_g128"}:
        from src.methods.mlx_vlm_baseline import (
            MLXBaselineConfig,
            quantize,
        )

        # Method name pins group_size; q_bits and friends can come from
        # YAML's ``mlx_vlm`` section (back-compat: empty section ⇒ defaults).
        group = int(method.split("g")[-1])
        raw_mlx_cfg = yaml_cfg.get("mlx_vlm", {}) or {}
        yaml_group = raw_mlx_cfg.get("q_group_size")
        if yaml_group is not None and int(yaml_group) != group:
            raise ValueError(
                f"mlx_vlm q_group_size={yaml_group} conflicts with method "
                f"{method}, which pins q_group_size={group}."
            )
        mlx_kwargs = _filter_dataclass_kwargs(MLXBaselineConfig, raw_mlx_cfg)
        # Method name pins group size so recipe names stay truthful.
        mlx_kwargs["q_group_size"] = group
        mlx_kwargs.setdefault("q_bits", 4)
        return quantize(merged_dir, output_dir, MLXBaselineConfig(**mlx_kwargs))
    if method == "unsloth_ud":
        from src.methods._stubs.unsloth_ud import UnslothUDConfig, quantize

        ud_kwargs = _filter_dataclass_kwargs(
            UnslothUDConfig, yaml_cfg.get("unsloth_ud", {})
        )
        return quantize(merged_dir, output_dir, UnslothUDConfig(**ud_kwargs))
    if method == "gptq":
        from src.methods.gptq import GPTQConfig, quantize

        gptq_kwargs = _filter_dataclass_kwargs(
            GPTQConfig, yaml_cfg.get("gptq", {})
        )
        return quantize(
            merged_dir,
            output_dir,
            GPTQConfig(**gptq_kwargs),
            plantnet_jsonl=extra.get("plantnet_jsonl"),
        )
    if method == "awq":
        from src.methods._stubs.awq import quantize

        return quantize(merged_dir, output_dir)
    if method == "bnb_nf4":
        from src.methods.bnb_nf4 import BnBNF4Config, quantize

        bnb_kwargs = _filter_dataclass_kwargs(
            BnBNF4Config, yaml_cfg.get("bnb_nf4", {})
        )
        # CLI override has the highest priority.
        skip = extra.get("bnb_skip_modules")
        if skip is not None:
            bnb_kwargs["skip_modules"] = skip
        return quantize(merged_dir, output_dir, BnBNF4Config(**bnb_kwargs))
    if method == "qat_export":
        from src.methods._stubs.qat_export import QATExportConfig, quantize

        qat_kwargs = _filter_dataclass_kwargs(
            QATExportConfig, yaml_cfg.get("qat_export", {}) or {}
        )
        if extra.get("qat_recipe_path") is not None:
            qat_kwargs["qat_recipe_path"] = extra["qat_recipe_path"]
        recipe = qat_kwargs.get("qat_recipe_path")
        if recipe is None:
            raise ValueError(
                "qat_export requires --qat_recipe_path <path-to-QAT-recipe.json>"
            )
        return quantize(merged_dir, output_dir, QATExportConfig(**qat_kwargs))
    raise ValueError(f"Unknown method: {method}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--method",
        required=False,
        default=None,
        choices=sorted(METHOD_REGISTRY),
        help=(
            "Quantization method to apply. Required unless --config "
            "provides a top-level ``method:`` field; if both are set, "
            "they must match."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config file with per-method overrides. Schema: "
            "top-level ``method:`` key + per-method sections (e.g. "
            "``gptq: {bits: 4, group_size: 128, ...}``). Fields unknown to "
            "the target dataclass are silently ignored."
        ),
    )
    parser.add_argument(
        "--base_model",
        default=None,
        help="HF repo id or local path of the bf16 base model. Required if --merged_dir not given.",
    )
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--merged_dir",
        type=Path,
        default=None,
        help="Pre-merged bf16 dir. Skips the LoRA merge step.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--qat_recipe_path",
        type=Path,
        default=None,
        help="Required for --method qat_export.",
    )
    parser.add_argument(
        "--plantnet_jsonl",
        type=Path,
        default=None,
        help="Domain calibration JSONL for --method gptq (optional).",
    )
    parser.add_argument(
        "--bnb_skip_modules",
        nargs="+",
        default=None,
        help=(
            "Substring skip list for --method bnb_nf4 — Linears whose "
            "dotted module name contains any of these are kept at bf16. "
            "E.g. --bnb_skip_modules embed_vision vision_tower. Used by "
            "the vision-collapse ablation."
        ),
    )
    parser.add_argument(
        "--merge_device",
        default="cpu",
        choices=("cpu", "cuda"),
        help=(
            "DEPRECATED. The safetensors-level merge runs pure-tensor "
            "work on CPU regardless of this flag. Kept for back-compat; "
            "non-default values log a warning and are otherwise ignored. "
            "Will be removed in a future round."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Step 0: resolve --config (optional) and --method (required by one
    # or the other source).
    yaml_cfg: dict = {}
    if args.config is not None:
        if not args.config.is_file():
            log.error("--config does not exist: %s", args.config)
            return 2
        import yaml as _yaml

        yaml_cfg = _yaml.safe_load(args.config.read_text()) or {}
        log.info("Loaded config: %s", args.config)

    cfg_method = yaml_cfg.get("method")
    if args.method is None and cfg_method is None:
        log.error(
            "--method is required (or set ``method:`` in --config). "
            "Known methods: %s",
            sorted(METHOD_REGISTRY),
        )
        return 2
    if args.method is not None and cfg_method is not None and args.method != cfg_method:
        log.error(
            "Conflict: --method=%s but --config sets method=%s. "
            "Either drop one or align them.",
            args.method, cfg_method,
        )
        return 2
    method = args.method or cfg_method
    if method not in METHOD_REGISTRY:
        log.error(
            "Unknown method %r. Known: %s",
            method, sorted(METHOD_REGISTRY),
        )
        return 2
    # CLI/YAML may have given a plantnet_jsonl in either spot; CLI wins.
    if args.plantnet_jsonl is None:
        cal = yaml_cfg.get("calibration", {}) or {}
        cal_pn = cal.get("plantnet_jsonl")
        if cal_pn:
            args.plantnet_jsonl = Path(cal_pn)

    # Step 1: produce a merged bf16 dir (or reuse one).
    #
    # Adapter-merge path goes through ``merge_safetensors.merge``
    # (safetensors-level LoRA merge) — NOT through HF transformers'
    # ``Gemma4ForConditionalGeneration.save_pretrained``. The HF path
    # silently drops ``k_proj`` / ``v_proj`` / ``k_norm`` / ``v_norm``
    # from KV-shared layers (Gemma 4 E2B layers 15-34) because those
    # tensors aren't registered ``nn.Parameter``s on the model class.
    # See ``scripts/merge_safetensors.py`` module docstring + AGENTS.md
    # known-bug list (transformers v5.8 + peft).
    if args.merged_dir:
        merged_dir = args.merged_dir
        if not merged_dir.is_dir():
            log.error("--merged_dir does not exist: %s", merged_dir)
            return 2
        log.info("Reusing existing merged dir: %s", merged_dir)
    elif args.adapter is not None:
        if not args.base_model:
            log.error("--adapter requires --base_model.")
            return 2
        if args.merge_device != "cpu":
            log.warning(
                "--merge_device=%s is ignored: the safetensors-level merge "
                "is pure-tensor work and runs on the host CPU regardless. "
                "Flag is kept for back-compat; will be removed in a future round.",
                args.merge_device,
            )
        from scripts.repair.merge_safetensors import (
            _resolve_base_dir,
            merge as merge_safetensors_merge,
        )

        merged_dir = args.output_dir.parent / f"{args.output_dir.name}.bf16-merged"
        if merged_dir.exists() and any(merged_dir.iterdir()):
            log.info("Reusing existing merged dir (non-empty): %s", merged_dir)
        else:
            log.info(
                "Merging adapter %s into base %s; output → %s",
                args.adapter, args.base_model, merged_dir,
            )
            base_dir = _resolve_base_dir(args.base_model)
            merge_safetensors_merge(base_dir, args.adapter, merged_dir)
    else:
        # No adapter and no pre-merged dir — quant directly off the base.
        # Resolving an HF repo id snapshot-downloads to the local cache;
        # a local path is returned as-is. Either way the dispatcher gets
        # a real directory to point its tooling at.
        if not args.base_model:
            log.error("Either --merged_dir or --base_model must be provided.")
            return 2
        from scripts.repair.merge_safetensors import _resolve_base_dir

        log.info(
            "No adapter to merge; resolving base %s and passing through.",
            args.base_model,
        )
        merged_dir = _resolve_base_dir(args.base_model)

    # Step 2: dispatch to the method.
    extra: dict = {"config_yaml": yaml_cfg}
    if args.qat_recipe_path:
        extra["qat_recipe_path"] = args.qat_recipe_path
    if args.plantnet_jsonl:
        extra["plantnet_jsonl"] = args.plantnet_jsonl
    if args.bnb_skip_modules:
        extra["bnb_skip_modules"] = list(args.bnb_skip_modules)
    try:
        final_dir = dispatch(method, merged_dir, args.output_dir, extra=extra)
    except ValueError as e:
        log.error("%s", e)
        return 2
    log.info("Done. Final output: %s", final_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
