"""QAT export — applies a QAT recipe at quantization time.

This is the export-side half of QAT. The training-side half (the
fake-quantization forward pass + bf16 backward) lives in
``finetune/`` because it requires modifying the SFT training loop.

## Policy note

QAT uses fake-quant in the forward pass and bf16 grads in the backward
pass. Strictly speaking it is NOT "4-bit training" — no 4-bit weights
or optimizer state ever exist. But the project policy in ``AGENTS.md``
says "no 8/4-bit training," which could be read either way.

**Status: AWAITING POLICY RULING.** Do not write QAT training code
until a teammate clears it.

## What this module does (when implemented)

After QAT-aware SFT produces a bf16 adapter PLUS a QAT recipe
metadata file (typically containing per-layer observer stats:
calibrated min/max, scales, zero points), this module:

1. Merges the QAT-trained adapter into the bf16 base (same as
   non-QAT export).
2. Applies the recorded scales/zero points to produce a 4-bit
   checkpoint where weight assignments come from QAT, not PTQ.
3. Routes the output through ``mlx_vlm.convert`` (or a custom MLX
   serializer if mlx_vlm refuses pre-quantized weights).

The expected accuracy improvement over PTQ baselines is largest when
the PTQ baselines themselves struggle — typically a few % on harder
benchmarks. Not worth the GPU days unless PTQ underperforms.

## Hardware

Mac with MLX (final conversion step). The bf16 merge can happen on
4090 / CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QATExportConfig:
    qat_recipe_path: Path | str
    target_bits: int = 4
    target_group_size: int = 64


def quantize(
    input_dir: Path | str,
    output_dir: Path | str,
    config: QATExportConfig,
):
    raise NotImplementedError(
        "QAT export pending: (1) policy ruling on whether QAT counts as "
        "4-bit training (it uses fake-quant fwd + bf16 backward), "
        "(2) QAT training-side code in finetune/."
    )
