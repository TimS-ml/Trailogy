"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant.

Wraps ``mlx_lm.quant.gptq.main``. Metal backend required. Do NOT run on
Linux/CUDA — the matching CUDA-targeted method is
``quantization/methods/gptq.py`` (gptqmodel-based).

LM-only: ``vision_tower`` + ``embed_vision`` + ``audio_tower`` are NOT
touched. To produce a deployable iOS bundle, vision modules need
re-attaching BF16 post-quantization (separate task).
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import register, warn_known_bug


@register("gptq")
def run_gptq(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run mlx_lm's GPTQ wrapper.

    cfg keys (under ``quant``):
        bits: int                  — quantized layer bit-width (default 4)
        group_size: int            — quantized layer group size (default 64)
        fallback_bits: int         — bit-width for non-GPTQ layers (default 4)
        fallback_group_size: int   — group size for non-GPTQ layers (default 64)
        num_samples: int           — calibration samples (default 256; -1 = all)
        sequence_length: int       — calibration sequence length (default 2048)
        seed: int                  — RNG seed (default 0)

    Idempotency: deletes any prior model.safetensors / index in
    ``output_dir`` before invoking the upstream main, so a stale partial
    output from a crashed run doesn't poison the new attempt.
    """
    warn_known_bug("gptq")
    output_dir.mkdir(parents=True, exist_ok=True)
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
        "--fallback-bits", str(q.get("fallback_bits", 4)),
        "--fallback-group-size", str(q.get("fallback_group_size", 64)),
        "--num-samples", str(q.get("num_samples", 256)),
        "--sequence-length", str(q.get("sequence_length", 2048)),
        "--seed", str(q.get("seed", 0)),
    ]

    from mlx_lm.quant import gptq as upstream

    t0 = time.time()
    _save_argv = sys.argv
    try:
        sys.argv = ["mlx_lm.quant.gptq", *argv]
        upstream.main()
    finally:
        sys.argv = _save_argv
    wall = time.time() - t0

    return {
        "method": "gptq",
        "upstream_module": "mlx_lm.quant.gptq",
        "argv": argv,
        "wall_seconds": wall,
    }
