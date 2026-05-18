"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant.

Wraps ``mlx_lm.quant.awq.main``. Metal backend required.

Known constraint as of mlx-lm 0.31.3: ``AWQ_MODEL_CONFIGS`` only covers
``llama``, ``mistral``, ``qwen2``, ``qwen3``, ``deepseek_v2``, ``gemma3``,
``gemma3_text``. Gemma 4 is **NOT** in the registry; this wrapper will
hit ``NotImplementedError`` at upstream line ~561 — the runner records
that as a stage failure and moves on. The Apple team would need to add
an AWQ_MODEL_CONFIGS entry for ``gemma4_text``/``gemma4`` before this
wrapper produces output.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from typing import Any

from . import read_model_type, register, warn_known_bug


def assert_supports_model(input_dir: Path) -> None:
    """Hard-fail before any model load if mlx_lm's AWQ wrapper has no
    entry for the input's ``model_type``.

    Saves the 30-60 s mlx_lm load + a crash deep in
    ``mlx_lm.quant.awq.main`` with an opaque KeyError. Raise instead
    with a roadmap-aware message.
    """
    model_type = read_model_type(input_dir)
    if model_type is None:
        return  # can't tell; let upstream decide
    try:
        from mlx_lm.quant.awq import AWQ_MODEL_CONFIGS
    except ImportError:
        return  # mlx_lm missing; the wrapper's own import will fail loudly
    if model_type not in AWQ_MODEL_CONFIGS:
        raise NotImplementedError(
            f"mlx_lm.quant.awq has no AWQ_MODEL_CONFIGS entry for "
            f"model_type={model_type!r}. Supported: "
            f"{sorted(AWQ_MODEL_CONFIGS.keys())}. "
            "The roadmap §'Why B.1 over B.2' calls out gemma4 as the "
            "missing entry that needs upstream patching; until that "
            "lands, this wrapper cannot produce output."
        )


@register("awq")
def run_awq(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run mlx_lm's AWQ wrapper.

    cfg keys (under ``quant``):
        bits: int                — main bit-width (default 4)
        group_size: int          — main group size (default 64)
        embed_bits: int          — embedding bit-width (default 4)
        embed_group_size: int    — embedding group size (default 32)
        num_samples: int         — calibration samples (default 128)
        sequence_length: int     — calibration sequence length (default 512)
        n_grid: int              — search grid size for per-channel scale (default 20)
        seed: int                — RNG seed (default 0)
    """
    warn_known_bug("awq")
    assert_supports_model(input_dir)
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
        "--embed-bits", str(q.get("embed_bits", 4)),
        "--embed-group-size", str(q.get("embed_group_size", 32)),
        "--num-samples", str(q.get("num_samples", 128)),
        "--sequence-length", str(q.get("sequence_length", 512)),
        "--n-grid", str(q.get("n_grid", 20)),
        "--seed", str(q.get("seed", 0)),
    ]

    from mlx_lm.quant import awq as upstream

    t0 = time.time()
    _save_argv = sys.argv
    try:
        sys.argv = ["mlx_lm.quant.awq", *argv]
        upstream.main()
    finally:
        sys.argv = _save_argv
    wall = time.time() - t0

    return {
        "method": "awq",
        "upstream_module": "mlx_lm.quant.awq",
        "argv": argv,
        "wall_seconds": wall,
    }
