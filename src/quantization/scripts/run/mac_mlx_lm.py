#!/usr/bin/env python3
"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant runner.

Resume-after-crash runner for one quantization variant. Reads a YAML
config, dispatches to one of ``src.methods.mac_mlx_lm.*``
wrappers, runs convert -> size -> smoke -> ppl in sequence with a
``state.json`` checkpoint after each stage.

Usage::

    python -m scripts.run.mac_mlx_lm \\
        --variant mac-mlx-lm-gptq-w4-g64 \\
        --config quantization/configs/mac-mlx-lm-gptq-w4-g64.yaml \\
        --input-dir  ~/work/gemma4-merged-bf16 \\
        --output-dir quantization/results/mac_mlx_lm/gptq-w4-g64

Resume: re-run the exact same command. Stages whose ``state.json``
status is ``done`` are skipped. A stage left in ``in_progress`` is
treated as a stale lock (process was killed) and re-run from scratch
(stages MUST be idempotent — they clear partial output before
running).

Skip / force a specific stage::

    --force-stage ppl     # re-run ppl even if done
    --skip-stage  ppl     # never run ppl in this invocation
    --only-stage  convert # run ONLY convert, skip everything else

The state machine itself is in ``src.common.resume``; it is
covered by unit tests in ``quantization/tests/test_resume.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a script even when not pip-installed: prepend the
# repo root so ``import src.* / scripts.*`` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml  # noqa: E402

from src.common.resume import (  # noqa: E402
    STATUS_DONE,
    Stage,
    StageOutcome,
    StateMachine,
)
from src.methods.mac_mlx_lm import METHOD_REGISTRY  # noqa: E402

log = logging.getLogger("mac_mlx_lm.runner")

# Pipeline stages, in order.
STAGES = ("convert", "size", "smoke", "ppl")

SMOKE_PROMPTS = [
    "Can you identify this species?",
    "What is the capital of France?",
    "If a hiker sees a tall tree with peeling white bark in the northeast US, what species is it likely to be?",
    "Write a one-sentence haiku about quantization.",
    "用一句话介绍 PlantNet 数据集。",
]


# ---------------------------------------------------------------------------
# Stage 1: convert — calls the method wrapper.
# ---------------------------------------------------------------------------

def stage_convert(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict:
    method_name = cfg["method"]
    fn = METHOD_REGISTRY.get(method_name)
    if fn is None:
        raise ValueError(
            f"Unknown method '{method_name}'. Registered: {sorted(METHOD_REGISTRY)}"
        )
    log.info("convert: dispatching to method=%s", method_name)
    return fn(cfg, input_dir, output_dir)


# ---------------------------------------------------------------------------
# Stage 2: size — bytes on disk + per-submodule.
# ---------------------------------------------------------------------------

def stage_size(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict:
    safetensors_files = sorted(output_dir.glob("model*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"No model.safetensors in {output_dir} — convert stage must run first."
        )

    total = 0
    per_file = {}
    for f in safetensors_files:
        sz = f.stat().st_size
        total += sz
        per_file[f.name] = sz

    # Per-submodule rollup from the safetensors header.
    import struct

    per_submodule: dict[str, int] = {}
    for f in safetensors_files:
        with open(f, "rb") as fh:
            (hdr_len,) = struct.unpack("<Q", fh.read(8))
            hdr = json.loads(fh.read(hdr_len))
        for name, info in hdr.items():
            if name == "__metadata__":
                continue
            a, b = info["data_offsets"]
            per_submodule[name.split(".")[0]] = (
                per_submodule.get(name.split(".")[0], 0) + (b - a)
            )

    result = {
        "total_bytes": total,
        "total_gb": total / 1e9,
        "per_file_bytes": per_file,
        "per_submodule_bytes": per_submodule,
        "per_submodule_gb": {k: v / 1e9 for k, v in per_submodule.items()},
    }
    (output_dir / "size.json").write_text(json.dumps(result, indent=2))
    log.info("size: %.3f GB total", result["total_gb"])
    return result


# ---------------------------------------------------------------------------
# Stage 3: smoke — load + 5 fixed prompts, save text.
# ---------------------------------------------------------------------------

def stage_smoke(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict:
    from mlx_lm import generate, load

    log.info("smoke: loading model from %s", output_dir)
    model, tokenizer = load(str(output_dir))

    smoke_cfg = cfg.get("smoke", {})
    max_tokens = smoke_cfg.get("max_tokens", 128)
    prompts = smoke_cfg.get("prompts", SMOKE_PROMPTS)

    lines: list[str] = []
    for i, p in enumerate(prompts):
        log.info("smoke: prompt %d/%d", i + 1, len(prompts))
        try:
            messages = [{"role": "user", "content": p}]
            text_in = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            text_in = p  # no chat template -> raw
        t0 = time.time()
        out = generate(model, tokenizer, prompt=text_in, max_tokens=max_tokens, verbose=False)
        dt = time.time() - t0
        lines.append(f"=== prompt {i + 1} ({dt:.1f}s) ===\nQ: {p}\nA: {out}\n")

    (output_dir / "smoke_generations.txt").write_text("\n".join(lines))
    log.info("smoke: wrote %d generations", len(prompts))
    return {"n_prompts": len(prompts)}


# ---------------------------------------------------------------------------
# Stage 4: ppl — WikiText-2 perplexity (chunked + resumable).
# ---------------------------------------------------------------------------

def stage_ppl(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict:
    """Streaming WikiText-2 NLL/token with batch-level resume.

    Persists ``ppl_partial.json`` every ``checkpoint_every`` batches with
    the running ``sum_nll`` + ``sum_tokens`` + ``batch_idx``. On re-run,
    resumes from the next batch.
    """
    import mlx.core as mx
    from mlx_lm import load

    ppl_cfg = cfg.get("ppl", {})
    batch_size = ppl_cfg.get("batch_size", 4)
    sequence_length = ppl_cfg.get("sequence_length", 512)
    max_batches = ppl_cfg.get("max_batches", 200)
    checkpoint_every = ppl_cfg.get("checkpoint_every", 32)

    partial_path = output_dir / "ppl_partial.json"
    if partial_path.exists():
        partial = json.loads(partial_path.read_text())
        start_batch = partial["batch_idx"] + 1
        sum_nll = partial["sum_nll"]
        sum_tokens = partial["sum_tokens"]
        log.info("ppl: resuming from batch %d (sum_nll=%.3f, sum_tokens=%d)",
                 start_batch, sum_nll, sum_tokens)
    else:
        start_batch = 0
        sum_nll = 0.0
        sum_tokens = 0

    if start_batch >= max_batches:
        log.info("ppl: already complete (start_batch=%d >= max_batches=%d)",
                 start_batch, max_batches)
    else:
        log.info("ppl: loading model from %s", output_dir)
        model, tokenizer = load(str(output_dir))

        # Tokenize wikitext-2 from the local mlx_lm load_data helper.
        from mlx_lm.quant.utils import load_data
        # load_data returns a list of tokenized sequences.
        data = load_data(tokenizer, num_samples=-1, sequence_length=sequence_length)
        log.info("ppl: %d sequences of length %d available", len(data), sequence_length)

        n_avail = len(data) // batch_size
        n_to_run = min(max_batches, n_avail) - start_batch

        for i in range(n_to_run):
            batch_idx = start_batch + i
            batch_start = batch_idx * batch_size
            batch = data[batch_start : batch_start + batch_size]
            tokens = mx.array(batch)
            # Standard causal LM NLL: shift inputs/labels.
            inputs = tokens[:, :-1]
            labels = tokens[:, 1:]
            logits = model(inputs)
            # Numerically-stable cross-entropy via mx.nn losses.
            import mlx.nn as nn
            losses = nn.losses.cross_entropy(logits, labels, reduction="none")
            mask = labels != tokenizer.pad_token_id if tokenizer.pad_token_id is not None else mx.ones_like(labels)
            sum_nll += float((losses * mask).sum())
            sum_tokens += int(mask.sum())
            mx.eval(logits)  # keep MLX from accumulating graph

            if (batch_idx + 1) % checkpoint_every == 0 or i == n_to_run - 1:
                partial_path.write_text(
                    json.dumps(
                        {
                            "batch_idx": batch_idx,
                            "sum_nll": sum_nll,
                            "sum_tokens": sum_tokens,
                        },
                        indent=2,
                    )
                )
                log.info("ppl: batch %d/%d ppl_so_far=%.3f",
                         batch_idx + 1, max_batches,
                         _safe_exp(sum_nll / max(1, sum_tokens)))

    nll_per_token = sum_nll / max(1, sum_tokens)
    ppl = _safe_exp(nll_per_token)
    result = {
        "perplexity": ppl,
        "nll_per_token": nll_per_token,
        "n_batches": min(max_batches, sum_tokens // (sequence_length - 1) // max(1, batch_size)) if sum_tokens else 0,
        "sum_tokens": sum_tokens,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "dataset": "wikitext-2 (mlx_lm.quant.utils.load_data default)",
    }
    (output_dir / "wikitext_ppl.json").write_text(json.dumps(result, indent=2))
    log.info("ppl: perplexity=%.3f over %d tokens", ppl, sum_tokens)
    return result


def _safe_exp(x: float) -> float:
    import math
    try:
        return math.exp(x)
    except OverflowError:
        return float("inf")


# ---------------------------------------------------------------------------
# Stage registry.
# ---------------------------------------------------------------------------

STAGE_FUNCS: dict[str, Stage] = {
    "convert": stage_convert,
    "size": stage_size,
    "smoke": stage_smoke,
    "ppl": stage_ppl,
}


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="Path to the bf16 merged model dir.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where the quantized model + eval artifacts land.")
    parser.add_argument("--force-stage", action="append", default=[],
                        help="Force this stage to re-run even if done. Repeatable.")
    parser.add_argument("--skip-stage", action="append", default=[],
                        help="Skip this stage entirely. Repeatable.")
    parser.add_argument("--only-stage",
                        help="Run ONLY this stage; all others skipped.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose == 0 else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = yaml.safe_load(args.config.read_text())
    args.input_dir = args.input_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log.info("variant=%s  method=%s", args.variant, cfg.get("method"))
    log.info("input_dir=%s", args.input_dir)
    log.info("output_dir=%s", args.output_dir)

    sm = StateMachine(
        state_path=args.output_dir / "state.json",
        variant=args.variant,
        stages=STAGES,
    )
    sm.load_or_init()

    # Determine which stages to actually run.
    if args.only_stage:
        to_run = (args.only_stage,)
    else:
        to_run = STAGES
    skip = set(args.skip_stage)
    force = set(args.force_stage)

    for stage in to_run:
        if stage in skip:
            log.info("[%s] skipped (--skip-stage)", stage)
            continue
        if stage in force:
            sm.reset_stage(stage)
        if sm.is_done(stage):
            log.info("[%s] done (cached) — skipping", stage)
            continue

        sm.mark_in_progress(stage)
        try:
            result = STAGE_FUNCS[stage](cfg, args.input_dir, args.output_dir)
            sm.mark_done(stage, result=result)
            log.info("[%s] DONE", stage)
        except Exception as exc:
            sm.mark_failed(stage, error=str(exc))
            log.exception("[%s] FAILED: %s", stage, exc)
            return 1

    log.info("all requested stages complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
