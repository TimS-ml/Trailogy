"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant.

Wraps ``mlx_lm.quant.dynamic_quant.main``. Metal backend required.

Dynamic quant runs a per-tensor sensitivity estimation pass (forward +
gradients on calibration data, accumulated in fp32 or bf16), then
picks a per-tensor bit-width assignment from a {low, high} pair such
that the *average* bits-per-weight matches ``target_bpw``. Less
expensive than DWQ but more than GPTQ/AWQ because of the gradient
pass for sensitivities.

Upstream writes ``<model_name>_sensitivities.json`` to the **current
working directory**. The wrapper changes cwd to ``output_dir`` before
calling main so the sensitivity file lands inside the variant dir, and
on resume we pass it back via ``--sensitivities`` to skip the
gradient pass.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

from . import register, warn_known_bug


@register("dynamic_quant")
def run_dynamic_quant(cfg: dict[str, Any], input_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run mlx_lm's dynamic_quant wrapper.

    cfg keys (under ``quant``):
        target_bpw: float        — target average bits per weight (default 4.0)
        low_bits: int            — bit-width for low-sensitivity tensors (default 3)
        low_group_size: int      — group size for low-bit tier (default 64)
        high_bits: int           — bit-width for high-sensitivity tensors (default 4)
        high_group_size: int     — group size for high-bit tier (default 64)
        grad_checkpoint: bool    — gradient checkpointing (default True)
        accumulation_dtype: str  — 'float32' | 'bfloat16' (default 'bfloat16' for 32 GB safety)
        seed: int                — RNG seed (default 0)
    """
    warn_known_bug("dynamic_quant")
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("model.safetensors", "model.safetensors.index.json"):
        p = output_dir / stale
        if p.exists():
            p.unlink()

    q = cfg.get("quant", {})

    # Sensitivity file caching: upstream writes
    # "<model_name>_sensitivities.json" to cwd. We `cd` into output_dir
    # so that file lands inside the variant dir; on a re-run we pass it
    # back with --sensitivities to skip the gradient pass.
    sensitivities_arg: list[str] = []
    # Match upstream's naming: model.replace('/', '_')
    sens_file_name = f"{str(input_dir).replace('/', '_')}_sensitivities.json"
    sens_path = output_dir / sens_file_name
    if sens_path.exists() and sens_path.stat().st_size > 0:
        sensitivities_arg = ["--sensitivities", str(sens_path)]

    argv = [
        "--model", str(input_dir),
        "--mlx-path", str(output_dir),
        "--target-bpw", str(q.get("target_bpw", 4.0)),
        "--low-bits", str(q.get("low_bits", 3)),
        "--low-group-size", str(q.get("low_group_size", 64)),
        "--high-bits", str(q.get("high_bits", 4)),
        "--high-group-size", str(q.get("high_group_size", 64)),
        "--accumulation-dtype", str(q.get("accumulation_dtype", "bfloat16")),
        "--seed", str(q.get("seed", 0)),
        *sensitivities_arg,
    ]
    if q.get("grad_checkpoint", True):
        argv.append("--grad-checkpoint")

    from mlx_lm.quant import dynamic_quant as upstream

    t0 = time.time()
    _save_argv = sys.argv
    _save_cwd = os.getcwd()
    try:
        os.chdir(output_dir)
        sys.argv = ["mlx_lm.quant.dynamic_quant", *argv]
        upstream.main()
    finally:
        sys.argv = _save_argv
        os.chdir(_save_cwd)
    wall = time.time() - t0

    return {
        "method": "dynamic_quant",
        "upstream_module": "mlx_lm.quant.dynamic_quant",
        "argv": argv,
        "wall_seconds": wall,
        "sensitivities_file": str(sens_path) if sens_path.exists() else None,
    }
