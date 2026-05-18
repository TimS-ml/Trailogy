"""HF GPTQModel → MLX QuantizedLinear bridge.

> Status: **placeholder + design spec**. ``bridge`` and ``naive_bridge``
> raise ``NotImplementedError``. See "Implementation plan" below.

## Why this exists

The team's quantization roadmap settles on **Route B.1**: train at bf16
on CUDA, post-train quantize via HF GPTQModel (mature OBS-style PTQ
with desc_act / LQER / dead-column handling that the Apple-side
``mlx_lm/quant/`` ports don't have), then bridge the HF-format output
into the MLX shape ``mlx_vlm.load`` consumes on iOS.

Today's status of the pieces:

- bf16 SFT merged checkpoint                                    ✅
- HF GPTQModel PTQ on CUDA (rows R1/R2, 68.4-68.8 % PlantNet)   ✅
- HF GPTQ → MLX bridge                                          ← THIS
- MLX bf16 reference / mlx_vlm.convert -q affine baselines      ✅

The bridge is the gate to a shipping iOS artifact. Without it the
calibration quality earned on the CUDA side stays trapped in
HF-format files mlx_vlm cannot read.

## Format gap

| Concern                  | HF GPTQModel output           | MLX QuantizedLinear expected      |
|--------------------------|-------------------------------|-----------------------------------|
| Weight storage           | I32 packed (8 nibbles / i32)  | U32 packed                        |
| Scales dtype             | F16                           | BF16                              |
| Per-group bias           | combined w/ scale (``qzeros``)| separate ``biases`` tensor        |
| ``desc_act=True``        | ``g_idx`` permutation per L   | not native; bake into weight cols |
| Tensor key naming        | ``model.layers.N.mlp.gate_proj.qweight`` | mlx-vlm-style ``language_model.model.layers.N.mlp.gate_proj.weight`` |
| ``config.json`` manifest | ``quantization_config: {bits, group_size, desc_act, ...}`` | ``quantization: {bits, group_size, mode}`` + per-tensor entries |

## Two implementation paths

### Path 1 — direct format translation (the real bridge)

Read every HF GPTQ tensor, transform packing + dtype + key naming,
emit MLX-format tensors. **Lossless**: preserves the OBS-derived
weight assignments exactly. Handles ``desc_act=True`` by reordering
the columns (``g_idx`` permutation) and re-packing in the new order.

Function: ``bridge(hf_gptq_dir, output_dir, *, baked_desc_act=True)``.

Expected on-disk size with vision/audio kept at bf16: ~3.6 GB.
Expected PlantNet val match: should equal R2's 68.8 % (the GPTQ
math is unchanged; only the storage format differs).

### Path 2 — dequant → bf16 → mlx_vlm.convert (naive baseline)

Use gptqmodel's own dequant path to materialize bf16 weights, save
them as a normal HF bf16 dir, then run ``mlx_vlm.convert -q``. This
**throws away GPTQ's Hessian-aware weight assignments** because
mlx_vlm.convert re-quantizes with a data-free affine recipe. The
output is "what does the worse-recipe re-quant of our better-recipe
weights look like" — useful as a sanity baseline (the real bridge
should beat it on PlantNet) but not the deliverable.

Function: ``naive_bridge(hf_gptq_dir, output_dir, *, q_bits=4, q_group_size=64)``.

## Implementation plan

1. Sketch ``bridge`` first as a pure-tensor refactor — operates on
   safetensors headers + numpy / torch arrays only. No CUDA needed.
   Should run on Linux 4090 or a Mac.
2. Test against a checked-in fixture (a ~50 MB hand-built HF GPTQ
   tensor and its expected MLX output). The roundtrip
   ``bridge → mlx_vlm.load → forward`` should match HF
   ``GPTQModel.from_quantized → forward`` within 1e-4 cosine on
   matched calibration prompts.
3. ``naive_bridge`` is implemented second as the comparison floor.
4. PlantNet eval on the bridged artifact validates that no accuracy
   is lost in the format translation.

## Why this file is here today

Pinning the design **before** the implementation so:

- New contributors can see where the work goes.
- The team's roadmap §B.1.3 has a code-side anchor.
- ``run_quant.py`` / ``run_eval.py`` can grow a path-1 / path-2 flag
  without breaking layout when the implementation lands.
"""

from __future__ import annotations

from pathlib import Path


def bridge(
    hf_gptq_dir: Path | str,
    output_dir: Path | str,
    *,
    baked_desc_act: bool = True,
) -> Path:
    """Translate an HF GPTQModel-quantized directory into the MLX
    QuantizedLinear format consumable by ``mlx_vlm.load``.

    Path 1 of the design above. Lossless w.r.t. GPTQ's calibration
    decisions — only the storage format changes.

    Args:
        hf_gptq_dir: Path to a directory written by
            ``gptqmodel.save_quantized`` (contains packed safetensors
            + ``quantize_config.json``).
        output_dir: Where to write the MLX-format artifact.
        baked_desc_act: When the HF input has ``desc_act=True``, bake
            the ``g_idx`` permutation into the weight column order so
            MLX (which doesn't support runtime permutation) can load
            the result without further help. Default True. Pass False
            only if you've validated the activation-order doesn't
            actually win on your target (R1 vs R2 in the team's
            ablation suggests ~0.5 pp drop).

    Returns:
        ``Path`` to ``output_dir`` (created).
    """
    raise NotImplementedError(
        "HF GPTQ → MLX bridge (B.1.3) is not implemented yet. See "
        "this module's docstring for the design + format-gap table, "
        "and the team's roadmap §B.1.3."
    )


def naive_bridge(
    hf_gptq_dir: Path | str,
    output_dir: Path | str,
    *,
    q_bits: int = 4,
    q_group_size: int = 64,
) -> Path:
    """Dequant HF GPTQ to bf16, then re-quantize via ``mlx_vlm.convert``.

    Path 2 of the design above. **Lossy** — throws away GPTQ's
    Hessian-aware weight assignments and re-quantizes with mlx_vlm's
    data-free affine recipe. Exists as the sanity baseline that the
    real bridge must beat on PlantNet match.

    Args:
        hf_gptq_dir: Path to the HF GPTQ output.
        output_dir: Where to write the MLX artifact.
        q_bits: Bit-width passed to ``mlx_vlm.convert``.
        q_group_size: Group size passed to ``mlx_vlm.convert``.

    Returns:
        ``Path`` to ``output_dir`` (created).
    """
    raise NotImplementedError(
        "Naive bridge (HF GPTQ → bf16 → mlx_vlm.convert -q) is not "
        "implemented yet. See this module's docstring for the design "
        "+ team's roadmap §B.1.3."
    )
