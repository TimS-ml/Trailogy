#!/usr/bin/env python3
"""Eval a quantized MLX model with a pre-computed EoRA adapter applied
at a given rank.

This is the eval-only half of ``eora_post_quant.py``: skip Phase 1
(adapter compute), load Phase 1's saved file, run PlantNet eval.

Usage:
    python -m scripts.run.eval_eora_only \\
        --quant-dir   results/<sft>/mlx_g64 \\
        --adapter     results/<sft>/eora/adapters_r64.safetensors \\
        --rank        64 \\
        --plantnet-jsonl ../finetune/data/english-desc/test.jsonl \\
        --plantnet-n  300 \\
        --output      results/<sft>/eora/eval_r64.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Importable: src/ from the quantization module root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("eval_eora_only")


def main(argv: list[str] | None = None) -> int:
    import mlx.core as mx
    from mlx_vlm import load as vlm_load
    from mlx_vlm import generate as _gen
    from mlx_vlm.prompt_utils import apply_chat_template as _chat

    from src.eval.plantnet import PlantNetConfig, run as plantnet_run
    from src.eval.model_loaders import ModelHandle
    from src.methods.eora_mlx import apply_adapters_from_file

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quant-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="EoRA adapter safetensors (from save_adapters).")
    parser.add_argument("--rank", type=int, default=None,
                        help="Truncate adapter to this rank. Default: use saved rank.")
    parser.add_argument("--plantnet-jsonl", type=Path, required=True)
    parser.add_argument("--plantnet-n", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output eval JSON path.")
    parser.add_argument("--variant-label", default=None,
                        help="Stamp variant string into eval.json (default: derived).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading quant model: %s", args.quant_dir)
    model, processor = vlm_load(str(args.quant_dir))

    log.info("Applying EoRA adapter: %s (rank=%s)", args.adapter, args.rank)
    n_patched = apply_adapters_from_file(
        model.language_model,
        str(args.adapter),
        rank=args.rank,
    )
    log.info("Patched %d layers.", n_patched)

    def infer(messages, image_path=None, max_new_tokens=128):
        # mlx_vlm 0.4.x apply_chat_template signature:
        #   (processor, config, prompt, add_generation_prompt=True,
        #    num_images=0, ...)
        prompt = _chat(
            processor,
            model.config,
            messages,
            num_images=1 if image_path else 0,
        )
        if image_path:
            result = _gen(
                model, processor, prompt,
                image=image_path,
                max_tokens=max_new_tokens,
                verbose=False,
            )
        else:
            result = _gen(
                model, processor, prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            )
        if hasattr(result, "text"):
            return result.text
        if isinstance(result, str):
            return result
        return str(result)

    handle = ModelHandle(
        model=model,
        processor=processor,
        backend="mlx_vlm+eora",
        infer_text=infer,
        device=str(mx.default_device()),
        model_dir=args.quant_dir,
    )
    cfg = PlantNetConfig(
        val_jsonl=args.plantnet_jsonl,
        n_samples=args.plantnet_n,
        max_new_tokens=128,
        seed=0,
    )

    t0 = time.perf_counter()
    result = plantnet_run(handle, cfg)
    elapsed = time.perf_counter() - t0

    label = args.variant_label or f"eora_r{args.rank}_on_{args.quant_dir.name}"
    payload = {
        "variant": label,
        "model_path": str(args.quant_dir),
        "adapter_path": str(args.adapter),
        "rank": args.rank,
        "n_patched": n_patched,
        "backend": "mlx_vlm+eora",
        "device": str(mx.default_device()),
        "benchmarks": {
            "plantnet_val": {
                "n": result.n,
                "species_match": result.species_match,
                "species_matches": result.species_matches,
                "rouge_l_mean": result.rouge_l_mean,
                "rouge_l_median": result.rouge_l_median,
                "avg_response_len": result.avg_response_len,
                "elapsed_s": elapsed,
            },
        },
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    log.info(
        "[%s] species_match=%.1f%% (%d/%d) ROUGE-L=%.3f elapsed=%.0fs → %s",
        label,
        result.species_match * 100,
        result.species_matches,
        result.n,
        result.rouge_l_mean,
        elapsed,
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
