#!/usr/bin/env python3
"""Post-quantization EoRA adapter for MLX quantized models.

Computes EoRA (arXiv:2410.21271) low-rank adapters that compensate for
quantization error, then evaluates at multiple ranks.

Usage:
    python -m scripts.run.eora_post_quant \
        --bf16-dir  quantization/results/_merged_bf16 \
        --quant-dir quantization/results/mlx_vlm_g64_sft_aug_enwiki \
        --output-dir quantization/results/eora_on_m2_g64 \
        --plantnet-jsonl finetune/data/english-desc/test.jsonl \
        --ranks 32 64 128 \
        --calib-samples 128 --calib-seq-len 512
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Ensure quantization/src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

log = logging.getLogger("eora_post_quant")


def load_calibration(processor, num_samples: int, seq_len: int):
    """Text calibration data from mlx_lm's built-in corpus."""
    import mlx.core as mx
    from mlx_lm.quant.utils import load_data

    tokenizer = processor.tokenizer
    log.info(
        "Loading calibration data: %d samples x %d tokens",
        num_samples, seq_len,
    )
    mx.random.seed(42)
    data = load_data(tokenizer, num_samples=num_samples, sequence_length=seq_len)
    log.info("Calibration shape: %s", data.shape)
    return data


def run_plantnet_eval(model, processor, val_jsonl: Path, n_samples: int = 300):
    """Run PlantNet species-match eval and return the result dict."""
    from mlx_vlm import generate as _gen
    from mlx_vlm.prompt_utils import apply_chat_template as _chat

    from src.eval.plantnet import PlantNetConfig, run as plantnet_run
    from src.eval.model_loaders import ModelHandle

    def infer(messages, image_path=None, max_new_tokens=128):
        prompt = _chat(processor, messages)
        if image_path:
            from PIL import Image
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

    import mlx.core as mx
    handle = ModelHandle(
        model=model,
        processor=processor,
        backend="mlx_vlm+eora",
        infer_text=infer,
        device=str(mx.default_device()),
        model_dir=str(val_jsonl.parent),
    )
    config = PlantNetConfig(
        val_jsonl=val_jsonl,
        n_samples=n_samples,
        max_new_tokens=128,
        seed=0,
    )
    return plantnet_run(handle, config)


def main(argv: list[str] | None = None) -> int:
    import mlx.core as mx

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--bf16-dir", type=Path, required=True,
                        help="Merged bf16 model directory")
    parser.add_argument("--quant-dir", type=Path, required=True,
                        help="Quantized model directory (e.g. M2 g64)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for adapters + eval")
    parser.add_argument("--plantnet-jsonl", type=Path, default=None,
                        help="test.jsonl for PlantNet eval (skip if unset)")
    parser.add_argument("--plantnet-n", type=int, default=300)
    parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--max-rank", type=int, default=128,
                        help="Max rank for adapter computation (compute once)")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load models + compute adapters
    # ------------------------------------------------------------------
    from mlx_vlm import load as vlm_load
    from src.methods.eora_mlx import (
        collect_xtx,
        compute_adapters,
        apply_adapters,
        save_adapters,
    )

    log.info("=== Phase 1: Loading quantized model ===")
    model, processor = vlm_load(str(args.quant_dir))

    log.info("=== Phase 1: Loading bf16 weights (lazy) ===")
    bf16_path = args.bf16_dir / "model.safetensors"
    if not bf16_path.exists():
        log.error("bf16 safetensors not found: %s", bf16_path)
        return 2
    bf16_weights = mx.load(str(bf16_path))
    log.info("bf16 weights: %d tensors loaded", len(bf16_weights))

    log.info("=== Phase 1: Calibration (X^T X collection) ===")
    calib_data = load_calibration(
        processor,
        num_samples=args.calib_samples,
        seq_len=args.calib_seq_len,
    )
    xtx_dict = collect_xtx(
        model.language_model,
        calib_data,
        batch_size=args.batch_size,
    )
    del calib_data
    mx.metal.clear_cache()

    log.info("=== Phase 1: Computing EoRA adapters (max_rank=%d) ===", args.max_rank)
    adapters = compute_adapters(
        model.language_model,
        bf16_weights,
        xtx_dict,
        max_rank=args.max_rank,
    )

    # Free calibration data and bf16 weights
    del xtx_dict, bf16_weights
    mx.metal.clear_cache()

    # Save adapters at max rank
    adapter_path = args.output_dir / "adapters_max.safetensors"
    save_adapters(adapters, str(adapter_path), rank=None)

    # Also save at each target rank
    for r in args.ranks:
        rank_path = args.output_dir / f"adapters_r{r}.safetensors"
        save_adapters(adapters, str(rank_path), rank=r)

    # Save metadata
    meta = {
        "quant_dir": str(args.quant_dir),
        "bf16_dir": str(args.bf16_dir),
        "max_rank": args.max_rank,
        "ranks": args.ranks,
        "calib_samples": args.calib_samples,
        "calib_seq_len": args.calib_seq_len,
        "n_layers": len(adapters),
        "layer_keys": sorted(adapters.keys()),
    }
    with open(args.output_dir / "eora_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("=== Phase 1 complete. Adapters saved to %s ===", args.output_dir)

    # ------------------------------------------------------------------
    # Phase 2: Evaluate at each rank
    # ------------------------------------------------------------------
    if args.plantnet_jsonl is None:
        log.info("No --plantnet-jsonl given; skipping eval.")
        return 0

    results_all: dict[str, dict] = {}

    for rank in sorted(args.ranks):
        log.info("=== Phase 2: Evaluating rank=%d ===", rank)

        # Reload model fresh for each rank (avoids stale wrappers)
        del model
        mx.metal.clear_cache()
        model, processor = vlm_load(str(args.quant_dir))

        n_patched = apply_adapters(model.language_model, adapters, rank=rank)
        log.info("Patched %d layers at rank=%d", n_patched, rank)

        t0 = time.perf_counter()
        result = run_plantnet_eval(
            model, processor,
            args.plantnet_jsonl,
            n_samples=args.plantnet_n,
        )
        elapsed = time.perf_counter() - t0

        result_dict = {
            "rank": rank,
            "n": result.n,
            "species_match": result.species_match,
            "species_matches": result.species_matches,
            "rouge_l_mean": result.rouge_l_mean,
            "rouge_l_median": result.rouge_l_median,
            "avg_response_len": result.avg_response_len,
            "elapsed_s": elapsed,
            "n_patched": n_patched,
        }
        results_all[f"r{rank}"] = result_dict

        log.info(
            "rank=%d: species_match=%.1f%% (%d/%d), ROUGE-L=%.3f, %.0fs",
            rank,
            result.species_match * 100,
            result.species_matches,
            result.n,
            result.rouge_l_mean,
            elapsed,
        )

        # Save per-rank eval
        eval_path = args.output_dir / f"eval_r{rank}.json"
        with open(eval_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    # Save combined results
    combined_path = args.output_dir / "eval_combined.json"
    with open(combined_path, "w") as f:
        json.dump(results_all, f, indent=2)

    log.info("=== All evaluations complete ===")
    for label, rd in sorted(results_all.items()):
        log.info(
            "  %s: %.1f%% (%d/%d)",
            label,
            rd["species_match"] * 100,
            rd["species_matches"],
            rd["n"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
