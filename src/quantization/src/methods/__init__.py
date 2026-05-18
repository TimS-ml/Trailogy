"""Quantization method implementations.

Each method module exposes a ``quantize(input_dir, output_dir, config)``
callable. Methods are intentionally decoupled: one method's failure
doesn't block another method's run.

Naming convention:
- ``mlx_vlm_baseline``: the current default (mlx_vlm.convert -q --q-bits 4)
- ``mlx_vlm_groups``: same convert, different ``--q-group-size``
- ``unsloth_ud``: replicate the Unsloth UD MLX 4-bit recipe
- ``gptq``: calibration-based PTQ via gptqmodel / auto-gptq
- ``awq``: Activation-aware Weight Quantization
- ``bnb_nf4``: bitsandbytes NF4 (reference only, not MLX-deployable)
- ``qat_export``: apply a QAT recipe at export time (paired with
  training-side code in ``finetune/`` if QAT is approved)
"""
