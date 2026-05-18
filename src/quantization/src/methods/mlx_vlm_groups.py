"""Group-size sweep around the MLX baseline.

Same ``mlx_vlm.convert`` invocation as the baseline, but with
``q_group_size`` varied. Smaller groups = more scales/biases (larger
file, better accuracy); larger groups = the inverse.

Useful for understanding the size/accuracy frontier of the
"default" MLX recipe before reaching for fancier methods.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .mlx_vlm_baseline import MLXBaselineConfig, quantize as _quantize_baseline

log = logging.getLogger(__name__)


# Group sizes worth measuring. 64 is the upstream default; 32 expands
# scales/biases ~2x in count; 128 contracts them ~2x. Anything below 32
# is rarely worth the file-size hit; anything above 128 starts to lose
# accuracy noticeably on small models.
GROUP_SIZE_GRID = (32, 64, 128)


@dataclass
class MLXGroupsConfig:
    q_bits: int = 4
    group_sizes: tuple[int, ...] = GROUP_SIZE_GRID


def quantize_grid(
    input_dir: Path, output_root: Path, config: MLXGroupsConfig | None = None
) -> dict[int, Path]:
    """Quantize the same input with each group size, writing per-size
    output dirs under ``output_root/g{N}/``.

    Returns a dict mapping group_size → output dir.
    """
    config = config or MLXGroupsConfig()
    output_root = Path(output_root)
    results: dict[int, Path] = {}
    for g in config.group_sizes:
        out_dir = output_root / f"g{g}"
        baseline_cfg = MLXBaselineConfig(
            quantize=True, q_bits=config.q_bits, q_group_size=g
        )
        log.info("Quantizing with group_size=%d → %s", g, out_dir)
        results[g] = _quantize_baseline(input_dir, out_dir, baseline_cfg)
    return results
