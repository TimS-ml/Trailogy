"""Baseline MLX quantization: ``mlx_vlm.convert -q --q-bits 4``.

This is what the existing ``finetune/src/export_mlx.py`` already does
end-to-end. This module re-exposes the convert step as a stand-alone
callable so the eval harness can compare it against alternative
methods on equal footing.

Hardware: Apple Silicon Mac with ``mlx`` and ``mlx_vlm`` installed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class MLXBaselineConfig:
    """Tunables for ``mlx_vlm.convert``.

    Most of these map directly to ``mlx_vlm.convert`` CLI flags. See
    https://github.com/Blaizzy/mlx-vlm for upstream docs.
    """

    quantize: bool = True
    q_bits: int = 4
    q_group_size: int = 64
    # Reserved for future per-module overrides — mlx_vlm doesn't expose
    # these uniformly today, but documenting the slot here.
    skip_quantize_keys: tuple[str, ...] = ()


def quantize(
    input_dir: Path,
    output_dir: Path,
    config: MLXBaselineConfig | None = None,
) -> Path:
    """Run ``mlx_vlm.convert`` on a bf16 multimodal Gemma 4 E2B
    safetensors directory and write the quantized output.

    Args:
        input_dir: Directory containing the merged bf16 model
            (output of ``finetune/src/export_mlx.py:merge_adapter`` or
            ``src.common.model_io.save_bf16_merged``).
        output_dir: Where to write the quantized MLX checkpoint.
        config: Quantization config; defaults to bits=4, group=64.

    Returns:
        The path to the directory containing the quantized
        safetensors + processor configs. Suitable for
        ``mlx_vlm.load`` and for the iOS bundle (after applying the
        processor_config.json patch — see
        ``finetune/src/export_mlx.py:patch_processor_config_for_mlx_swift``).
    """
    config = config or MLXBaselineConfig()
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    # Best-effort import check so we fail with a clear message on a
    # non-Mac box rather than mid-subprocess.
    try:
        import mlx_vlm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "mlx_vlm is not installed. This method requires an Apple "
            "Silicon Mac with mlx + mlx_vlm. To run, install via "
            "`pip install mlx mlx_vlm` on a Mac."
        ) from e

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "mlx_vlm.convert",
        "--hf-path",
        str(input_dir),
        "--mlx-path",
        str(output_dir),
    ]
    if config.quantize:
        cmd += ["-q", "--q-bits", str(config.q_bits)]
        # mlx_vlm.convert exposes --q-group-size in newer versions; older
        # versions only accept --q-bits. Forward it if our config asks
        # for a non-default; the subprocess will error visibly if the
        # installed mlx_vlm doesn't support it.
        if config.q_group_size != 64:
            cmd += ["--q-group-size", str(config.q_group_size)]
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(
            "mlx_vlm.convert failed (exit %d)\nstdout: %s\nstderr: %s",
            result.returncode,
            result.stdout,
            result.stderr,
        )
        raise RuntimeError("mlx_vlm.convert returned non-zero")
    if result.stdout.strip():
        log.info("mlx_vlm output:\n%s", result.stdout.strip())
    return output_dir
