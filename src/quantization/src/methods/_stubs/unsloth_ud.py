"""Unsloth Dynamic (UD) MLX 4-bit replication — STUB + INVESTIGATION.

Unsloth's "UD" branding marks their mixed-precision quantization
recipe: most layers go to 4-bit, but a subset of "sensitive" layers
(typically ``lm_head``, ``embed_tokens``, certain attention proj
heads) are kept at higher precision.

There are two reference models on HF, both `unsloth/gemma-4-E2B-it-UD-MLX-4bit`:
- HEAD (4.52 GB) — current recipe, over our 4 GB ceiling.
- Commit ``9ee11f5`` (3.55 GB) — older recipe, under our ceiling.

Goal of this module: replicate the ``9ee11f5`` recipe from the bf16
base, so we can ship a quantized SFT'd model under 4 GB with the same
philosophy.

## Implementation status

**NOT IMPLEMENTED.** This is a stub.

Open questions before any code is written:

1. **Where is the UD recipe defined?**
   Unsloth's quantization code lives in ``unslothai/unsloth`` under
   ``unsloth/save.py`` and ``unsloth/models/_utils.py`` (subject to
   refactor). The MLX-specific path goes through their
   ``unsloth/save_to_mlx`` or similar — needs source inspection at the
   ``9ee11f5`` commit timestamp of the HF release.

2. **Does the recipe require Unsloth training-loop integration, or
   can it be applied to a vanilla HF bf16 checkpoint?**
   The UD recipe in their text-only releases is config-driven and
   doesn't require Unsloth-specific training. The MLX VLM recipe is
   newer and may have hidden dependencies. The HF model card for
   ``unsloth/gemma-4-E2B-it-UD-MLX-4bit`` should clarify.

3. **Why did 9ee11f5 → HEAD grow from 3.55 GB → 4.52 GB?**
   Diff their two ``model.safetensors`` headers tensor-by-tensor.
   Whichever tensors changed dtype from int4 → bf16/fp16 are the
   "promoted" ones — that's the diff that costs ~1 GB.

## Plan

Once the open questions resolve:

1. Read ``9ee11f5`` model.safetensors header → enumerate per-tensor
   dtypes → produce a "promotion list" of tensors kept above 4-bit.
2. Apply that promotion list at quantization time. Likely path:
   patch ``mlx_vlm.convert`` to skip-quantize the promoted tensors
   (the upstream CLI accepts `--skip-quantize-keys` regex or similar
   in newer versions — check).
3. If `mlx_vlm.convert` lacks per-tensor opt-out, drop to the
   ``mlx`` Python API directly: call ``mx.quantize`` on the non-promoted
   tensors and write the rest as fp16 / bf16.

## Hardware

Apple Silicon Mac (same as ``mlx_vlm_baseline``).

## Why this is a stub today

The investigation step (read Unsloth source, diff the two HF releases)
is the gate. We don't write code against an uncertain target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UnslothUDConfig:
    """Configuration for replicating an Unsloth UD MLX recipe.

    ``promotion_keys``: list of safetensors key substrings whose
    tensors should be kept at higher precision (typically fp16) rather
    than quantized to 4-bit. Populated by diffing two HF releases.
    """

    q_bits: int = 4
    q_group_size: int = 64
    promotion_keys: tuple[str, ...] = field(default_factory=tuple)
    promoted_dtype: str = "float16"


def quantize(
    input_dir: Path,
    output_dir: Path,
    config: UnslothUDConfig | None = None,
) -> Path:
    """Apply an Unsloth-UD-style mixed-precision quantization."""
    raise NotImplementedError(
        "Unsloth UD replication not implemented. See module docstring "
        "for the investigation plan. Start by running "
        "`scripts/diff_unsloth_ud.py` to extract the promotion list "
        "from the public HF releases."
    )
