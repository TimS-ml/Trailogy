"""Cross-format bridges between PTQ outputs and inference runtimes.

A "bridge" takes the output of one quantization toolchain (e.g.
HF-format gptqmodel) and produces an artifact a different runtime
(e.g. mlx_vlm) can consume — without re-running the underlying
calibration math. This is distinct from ``methods/`` (which produces
the original quantized artifact) and ``eval/`` (which loads + scores).

Bridges currently planned (priorities per the team's quantization
roadmap):

- ``hf_gptq_to_mlx``: HF GPTQModel safetensors → MLX QuantizedLinear
  format. **The B.1.3 deliverable.** Not implemented; see module
  docstring for the design.
"""
