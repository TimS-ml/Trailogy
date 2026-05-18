#!/usr/bin/env python3
"""Microbenchmark for gptqmodel kernel backends.

Loads the same on-disk GPTQ checkpoint under N different backends and
times generation on a small fixed sample of PlantNet val. Used to decide
whether switching from Triton → Marlin / ExLlamaV2 / Machete is worth
re-running a 3-hour full eval.

What it measures:
    * tokens / second (averaged over the sampled images)
    * mean wall-clock per image (load is excluded; only generation)
    * top-1 sanity: did each backend produce the same species_match
      number on this small sample? (the dequant math is mathematically
      identical; differing accuracy across backends is a load-time bug)

What it does NOT measure:
    * model load time (Marlin pays a one-time JIT cost on first load)
    * end-to-end eval wall (use --plantnet_n in run_eval.py for that)
    * quality on a statistically-meaningful sample (default n=20 is for
      timing only)

Example:
    python -m scripts.inspect.benchmark_gptq_backend \\
        --model_dir quantization/results/gptq_w4g128_da0 \\
        --plantnet_val_jsonl finetune/data/val.jsonl \\
        --backends triton marlin exllama_v2 \\
        --n_samples 20

Notes on backend / checkpoint compatibility:
    * Marlin classic requires sym=True, group ∈ {32, 64, 128}, desc_act=False.
      The da=0 checkpoint matches; the da=1 (desc_act=True) checkpoint
      does NOT — pick exllama_v2 for da=1.
    * Machete is Hopper/Ada specific; on 4090 it works, on older GPUs it
      will fall back to torch and give no speedup.
    * Triton works for everything but is the slowest at batch=1 decode.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.model_loaders import (  # noqa: E402
    GPTQ_BACKEND_CHOICES,
    load_hf_gptq,
)

log = logging.getLogger(__name__)


def _read_jsonl(path: Path, n: int, seed: int) -> list[dict]:
    """Read N records from a JSONL file with a fixed seed for reproducibility."""
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:n]


def _first_user_message(rec: dict) -> dict | None:
    """Extract the first user-turn message dict from a PlantNet val record."""
    conv = rec.get("conversations") or rec.get("messages") or []
    for msg in conv:
        if msg.get("role") == "user":
            return msg
    return None


def _image_path_from_record(rec: dict) -> str | None:
    """PlantNet val records embed the image path inside the user content blocks."""
    msg = _first_user_message(rec)
    if not msg:
        return None
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                return block.get("image") or block.get("path")
    return None


def _benchmark_one(
    model_dir: Path,
    backend: str,
    samples: list[dict],
    max_new_tokens: int,
) -> dict:
    """Load the GPTQ model under one backend and time generation on `samples`."""
    log.info("=== backend=%s ===", backend)
    t_load_start = time.perf_counter()
    handle = load_hf_gptq(model_dir, backend=backend)
    t_load_end = time.perf_counter()
    log.info("  load: %.1fs", t_load_end - t_load_start)

    per_sample_times: list[float] = []
    per_sample_tokens: list[int] = []
    predictions: list[str] = []

    for i, rec in enumerate(samples):
        msg = _first_user_message(rec)
        if msg is None:
            log.warning("  sample %d: no user message, skipping", i)
            continue
        image_path = _image_path_from_record(rec)
        t0 = time.perf_counter()
        out = handle.infer_text(
            messages=[msg],
            image_path=image_path,
            max_new_tokens=max_new_tokens,
        )
        t1 = time.perf_counter()
        per_sample_times.append(t1 - t0)
        # Approx token count: word-split is fine for relative comparison.
        per_sample_tokens.append(len(out.split()))
        predictions.append(out)
        log.info("  [%d/%d] %.2fs (~%d toks)", i + 1, len(samples), t1 - t0, per_sample_tokens[-1])

    total_time = sum(per_sample_times)
    total_tokens = sum(per_sample_tokens)
    return {
        "backend": backend,
        "load_seconds": round(t_load_end - t_load_start, 2),
        "n_samples": len(per_sample_times),
        "total_seconds": round(total_time, 2),
        "mean_seconds_per_sample": round(total_time / max(1, len(per_sample_times)), 3),
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / max(1e-6, total_time), 2),
        # Keep the first 32 chars of each prediction so we can spot-check
        # that different backends produce ~identical outputs (sampling is
        # off by default so they should be bit-exact, but FP rounding
        # differences across kernels can shift a token).
        "prediction_previews": [p[:64] for p in predictions],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--plantnet_val_jsonl", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["triton", "marlin"],
        choices=GPTQ_BACKEND_CHOICES,
        help="Which gptqmodel backends to A/B time.",
    )
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="If given, write per-backend timing as JSON.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.model_dir.is_dir():
        parser.error(f"--model_dir {args.model_dir} not found")
    if not args.plantnet_val_jsonl.is_file():
        parser.error(f"--plantnet_val_jsonl {args.plantnet_val_jsonl} not found")

    samples = _read_jsonl(args.plantnet_val_jsonl, args.n_samples, args.seed)
    log.info("Loaded %d PlantNet samples for benchmarking.", len(samples))

    results = []
    for backend in args.backends:
        try:
            r = _benchmark_one(args.model_dir, backend, samples, args.max_new_tokens)
            results.append(r)
        except Exception as e:  # noqa: BLE001
            log.exception("backend=%s failed: %s", backend, e)
            results.append({"backend": backend, "error": repr(e)})

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"{'backend':<14} {'load(s)':>8} {'mean(s)':>9} {'tok/s':>8} {'n':>4}")
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<14} ERROR: {r['error'][:60]}")
            continue
        print(
            f"{r['backend']:<14} {r['load_seconds']:>8.1f} "
            f"{r['mean_seconds_per_sample']:>9.3f} "
            f"{r['tokens_per_second']:>8.1f} {r['n_samples']:>4}"
        )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
