"""Quantization-method **stubs** — placeholders that raise
``NotImplementedError`` when invoked.

Three methods live here today, all blocked on external work per the
team's quantization roadmap:

- ``awq``        — CUDA-side AWQ. Lower priority than GPTQ; the
                   roadmap dismisses AWQ as a sustained-effort track
                   (still useful as a research comparison).
- ``unsloth_ud`` — Replicate Unsloth's UD MLX-4bit recipe at commit
                   9ee11f5 (3.55 GB). Gated on extracting the
                   promotion-key list from the public HF release
                   (``scripts/diff_unsloth_ud.py``).
- ``qat_export`` — QAT export-time step. Gated on (a) policy ruling
                   on whether QAT counts as "4-bit training" under
                   ``AGENTS.md [0]``, and (b) the QAT training-loop
                   code in ``finetune/`` that doesn't exist yet.

Each stub still appears in ``METHOD_REGISTRY`` so ``run_quant.py
--method <name>`` dispatches here and raises a clear
``NotImplementedError`` rather than silently no-op'ing. Tests in
``test_run_quant_dispatch.py`` pin that contract.

Moving these out of the top-level ``methods/`` directory keeps the
"what's real" vs "what's planned" distinction visible at a glance —
``ls methods/`` shows only methods that can actually produce a
quantized artifact today.
"""
