"""Shared utilities for quantization methods.

Submodules:

- ``safetensors_io``: read safetensors headers without loading tensor
  data; enumerate keys; per-tensor and per-bucket size reporting.
- ``model_io``: bf16 base + adapter loading (merge_and_unload), with
  the same vision-tower-preserving safety as ``export_mlx.py``.
- ``sizing``: directory-level size accounting, formatted reports.
"""
