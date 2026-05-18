"""Eval harness for quantization variants.

Design:

- Every variant produces the SAME JSON output schema (see
  ``runner.py`` docstring + ``eval/README.md``).
- Per-benchmark runners take a "model handle" produced by a
  ``model_loader`` callable. The loader hides whether we're running
  bf16 HF (CUDA) or INT4 MLX (Apple Silicon) — the benchmarks don't
  care.
- One orchestrator (``runner.run_all``) sweeps through the chosen
  benchmarks and aggregates one JSON file per variant.

Submodules:

- ``model_loaders``: lazy loaders for bf16 HF and INT4 MLX checkpoints.
- ``plantnet``: PlantNet val species-match (the domain metric).
- ``wikitext_ppl``: WikiText-103 perplexity (catastrophic-language guard).
- ``vqav2``: VQAv2 dev-test accuracy (broader VLM metric).
- ``runner``: orchestrates the above and writes the JSON output.
"""
