"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant.

Wraps ``mlx_lm.quant.dwq.main``. Metal backend required.

DWQ runs a **distillation training loop**: the teacher is the bf16
model, the student is the quantized model, and KL divergence between
their logits is minimized. This is the longest-running of the four
methods (hours on a multi-billion-param model). DWQ has the most
hyperparameters of the four; check ``mlx_lm/quant/dwq.py`` for the
full list — only the safe-to-tune knobs are surfaced here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from . import register, warn_known_bug


@register("dwq")
def run_dwq(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run mlx_lm's DWQ wrapper.

    cfg keys (under ``quant``):
        bits: int             — bit-width (default 4)
        group_size: int       — group size (default 64)
        num_samples: int      — training samples (default 1024 — half upstream default)
        max_seq_length: int   — training seq length (default 1025)
        learning_rate: float  — optimizer LR (default 1e-6)
        batch_size: int       — training batch size (default 2 for 32 GB RAM safety)
        data_path: str        — HF dataset (default 'allenai/tulu-3-sft-mixture')
        grad_checkpoint: bool — enable gradient checkpointing (default True for 32 GB safety)
        seed: int             — RNG seed (default 0)

    The wrapper passes ``--target-dir <output_dir>/dwq_targets`` so the
    precomputed teacher logits cache survives across resumes (this is
    the expensive part of DWQ — recomputing it on every resume would
    defeat the point of resume).
    """
    warn_known_bug("dwq")
    output_dir.mkdir(parents=True, exist_ok=True)
    target_dir = output_dir / "dwq_targets"
    target_dir.mkdir(parents=True, exist_ok=True)

    for stale in ("model.safetensors", "model.safetensors.index.json"):
        p = output_dir / stale
        if p.exists():
            p.unlink()

    q = cfg.get("quant", {})
    argv = [
        "--model", str(input_dir),
        "--mlx-path", str(output_dir),
        "--bits", str(q.get("bits", 4)),
        "--group-size", str(q.get("group_size", 64)),
        "--num-samples", str(q.get("num_samples", 1024)),
        "--max-seq-length", str(q.get("max_seq_length", 1025)),
        "--learning-rate", str(q.get("learning_rate", 1e-6)),
        "--batch-size", str(q.get("batch_size", 2)),
        "--data-path", str(q.get("data_path", "allenai/tulu-3-sft-mixture")),
        "--target-dir", str(target_dir),
        "--seed", str(q.get("seed", 0)),
    ]
    if q.get("grad_checkpoint", True):
        argv.append("--grad-checkpoint")

    from mlx_lm.quant import dwq as upstream

    t0 = time.time()
    _save_argv = sys.argv
    try:
        sys.argv = ["mlx_lm.quant.dwq", *argv]
        upstream.main()
    finally:
        sys.argv = _save_argv
    wall = time.time() - t0

    return {
        "method": "dwq",
        "upstream_module": "mlx_lm.quant.dwq",
        "argv": argv,
        "wall_seconds": wall,
    }
