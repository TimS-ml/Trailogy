"""[RUNTIME] Mac (Apple Silicon, M-series) — mlx-lm native quant.

Wrappers around `mlx_lm/quant/<method>.py` for use from the
production quantization pipeline. Metal backend required. The CUDA
counterparts live one level up in ``quantization/methods/``.

See ``quantization/scripts/run_mac_mlx_lm.py`` for the runner with
resume-after-crash semantics.

NOTE (2026-05-14): the wrappers below currently call
``mlx_lm.quant.<method>.main()``, which loads Gemma 4 via mlx-lm's
model class. That class diverges from the iOS-runtime mlx-vlm class
(RMSNorm vs RMSNormZeroShift, etc.), so the produced outputs are not
directly deployable. A rewrite to use the hybrid flow (mlx_vlm.load +
mlx_lm.quant.* core + mlx_vlm save) is queued.

## Known upstream bugs

Per the team's quantization roadmap §"Why B.1 over B.2", each mlx_lm
method has an unresolved upstream issue on Gemma 4:

- ``gptq``  — NaN logits on Gemma 4 (mlx-lm round 2026-05-13).
              Fix requires the hybrid flow + algorithm-level ports
              (act-order, dead-column, auto-clip, LQER) that the
              CUDA-side ``gptqmodel`` already ships.
- ``awq``   — ``AWQ_MODEL_CONFIGS`` has no ``gemma4`` entry; convert
              fails at upstream ``mlx_lm/quant/awq.py:~561``.
              Preflight in ``mac_mlx_lm.awq.assert_supports_model``.
- ``dwq``   — broadcast bug at upstream line 113 in the validation-
              loss path; reproduces on Gemma 4 regardless of the
              forward-pass model tree.
- ``dynamic_quant`` — runs end-to-end but is LM-only; needs
              ``scripts/splice_lm_into_multimodal.py`` to re-attach
              the bf16 vision_tower + embed_vision before
              ``mlx_vlm.load`` accepts the result.

Each wrapper logs its entry from this list when invoked, so a fresh
operator sees the roadmap context without grepping for it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Method registry — wrappers register themselves so the runner can
# look them up by config-declared name.
METHOD_REGISTRY: dict[str, "WrapperFn"] = {}


def register(name: str):
    """Decorator: register a wrapper under ``name``."""

    def _decorator(fn):
        METHOD_REGISTRY[name] = fn
        return fn

    return _decorator


# Each wrapper is a callable: (cfg, input_dir, output_dir) -> dict
WrapperFn = "Callable[[dict, pathlib.Path, pathlib.Path], dict]"


# ---------------------------------------------------------------------------
# Preflight helpers — used by individual wrappers
# ---------------------------------------------------------------------------


# Roadmap-aware known-bug strings. Keyed by wrapper name. Each wrapper
# logs its entry once at invocation time.
KNOWN_UPSTREAM_BUGS: dict[str, str] = {
    "gptq": (
        "mlx_lm.quant.gptq produced NaN logits on Gemma 4 in the "
        "2026-05-13 round; output may be unusable. The roadmap "
        "prioritizes the CUDA-side gptqmodel + B.1.3 bridge instead "
        "of the mlx-lm port."
    ),
    "awq": (
        "mlx_lm.quant.awq has no gemma4 entry in AWQ_MODEL_CONFIGS. "
        "Convert is expected to fail at upstream awq.py:~561 until "
        "Apple adds the model_type registration."
    ),
    "dwq": (
        "mlx_lm.quant.dwq has a broadcast bug at upstream line 113 "
        "in the validation-loss path; reproduces on Gemma 4 "
        "regardless of forward-pass tree."
    ),
    "dynamic_quant": (
        "mlx_lm.quant.dynamic_quant runs end-to-end but is LM-only. "
        "Re-attach bf16 vision via "
        "scripts/splice_lm_into_multimodal.py before mlx_vlm.load."
    ),
}


def warn_known_bug(wrapper_name: str) -> None:
    """Log the entry from ``KNOWN_UPSTREAM_BUGS`` if one exists."""
    msg = KNOWN_UPSTREAM_BUGS.get(wrapper_name)
    if msg:
        log.warning("[%s] known upstream bug: %s", wrapper_name, msg)


def read_model_type(input_dir: Path) -> str | None:
    """Return ``config.json``'s ``model_type``, or ``None`` if unreadable."""
    config = input_dir / "config.json"
    if not config.is_file():
        return None
    try:
        return json.loads(config.read_text()).get("model_type")
    except Exception:  # noqa: BLE001
        return None


# Import wrappers so the @register side-effects fire.
from . import awq  # noqa: E402,F401
from . import dwq  # noqa: E402,F401
from . import dynamic_quant  # noqa: E402,F401
from . import gptq  # noqa: E402,F401
