"""AWQ — Activation-aware Weight Quantization. STUB, low priority.

AWQ scales weights by activation magnitudes before rounding to 4-bit,
preserving accuracy on activation outliers (which dominate quant error
in transformer MLPs).

Defer until GPTQ smoke clarifies multimodal Gemma 4 backend support.
The two methods share the same calibration-data + sequential-pass
structure; whatever we learn from GPTQ applies here.

Hardware: NVIDIA CUDA.

Implementation status: not started. Slot reserved.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AWQConfig:
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True


def quantize(*args, **kwargs):  # noqa: D401, ANN001, ANN201
    raise NotImplementedError(
        "AWQ not implemented yet. Priority is below GPTQ — defer until "
        "GPTQ smoke clarifies multimodal Gemma 4 backend support."
    )
