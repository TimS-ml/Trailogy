"""Directory and per-submodule size accounting.

Built on top of ``safetensors_io``. Used by:

- ``scripts/inspect_quantized.py`` to print a human-readable size report.
- ``scripts/compare_sizes.py`` to diff two quantized checkpoints.
- The full eval matrix output, where each variant logs its on-disk size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .safetensors_io import (
    TensorEntry,
    bucket_by_submodule,
    enumerate_directory,
    summarize_size,
)


def fmt_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.1f} KB"
    return f"{n} B"


@dataclass
class DirectorySize:
    """On-disk size accounting for one model directory."""

    directory: Path
    total_safetensors_bytes: int
    per_submodule_bytes: dict[str, int]
    per_submodule_avg_bits: dict[str, float]
    dtype_histogram: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    non_safetensors_bytes: int = 0

    @property
    def total_disk_bytes(self) -> int:
        return self.total_safetensors_bytes + self.non_safetensors_bytes

    def format_report(self) -> str:
        lines = [
            f"Directory: {self.directory}",
            f"  Total on disk:        {fmt_bytes(self.total_disk_bytes)}",
            f"    safetensors:        {fmt_bytes(self.total_safetensors_bytes)}",
            f"    other files:        {fmt_bytes(self.non_safetensors_bytes)}",
            "",
            "  Per submodule (safetensors only):",
        ]
        for sub, nb in self.per_submodule_bytes.items():
            avg_bits = self.per_submodule_avg_bits.get(sub, 0.0)
            lines.append(
                f"    {sub:18s}  {fmt_bytes(nb):>10s}   avg {avg_bits:5.2f} bits/elt"
            )
        if self.dtype_histogram:
            lines.append("")
            lines.append("  Dtype histogram:")
            for dt, (n_tensors, nb, numel) in self.dtype_histogram.items():
                avg = nb * 8.0 / numel if numel else 0.0
                lines.append(
                    f"    {dt:12s}  {n_tensors:4d} tensors  {fmt_bytes(nb):>10s}   avg {avg:5.2f} bits/elt"
                )
        return "\n".join(lines)


def measure_directory(directory: Path) -> DirectorySize:
    """Walk a model directory, sum safetensors bytes (+ any other
    non-safetensors file bytes), and bucket by submodule.

    Non-safetensors bytes include config.json, tokenizer files, etc.
    Those are usually small (<32 MB total) but worth including in
    "total on disk" because the iOS bundle needs them all.
    """
    directory = Path(directory)
    entries = enumerate_directory(directory)
    grouped = bucket_by_submodule(entries)

    per_submodule_bytes: dict[str, int] = {}
    per_submodule_avg_bits: dict[str, float] = {}
    for sub, sub_entries in grouped.items():
        summary = summarize_size(sub_entries)
        per_submodule_bytes[sub] = summary["total_bytes"]
        per_submodule_avg_bits[sub] = summary["avg_bits_per_element"]

    full = summarize_size(entries)

    # Count non-safetensors bytes (config, tokenizer, etc.).
    non_st = 0
    for f in directory.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix == ".safetensors":
            continue
        try:
            non_st += f.stat().st_size
        except OSError:
            pass

    return DirectorySize(
        directory=directory,
        total_safetensors_bytes=full["total_bytes"],
        per_submodule_bytes=per_submodule_bytes,
        per_submodule_avg_bits=per_submodule_avg_bits,
        dtype_histogram=full["dtype_histogram"],
        non_safetensors_bytes=non_st,
    )


def diff_directories(
    a: DirectorySize, b: DirectorySize, label_a: str = "A", label_b: str = "B"
) -> str:
    """Side-by-side per-submodule size diff. Useful for understanding
    why one method produces 3.58 GB and another 4.52 GB.
    """
    lines = [
        f"  {'submodule':18s}  {label_a:>14s}  {label_b:>14s}  {'delta':>14s}",
        "  " + "-" * 64,
    ]
    keys = sorted(set(a.per_submodule_bytes) | set(b.per_submodule_bytes))
    for k in keys:
        sa = a.per_submodule_bytes.get(k, 0)
        sb = b.per_submodule_bytes.get(k, 0)
        delta = sb - sa
        lines.append(
            f"  {k:18s}  {fmt_bytes(sa):>14s}  {fmt_bytes(sb):>14s}  "
            f"{('+' if delta >= 0 else '-')}{fmt_bytes(abs(delta)):>13s}"
        )
    total_a = a.total_disk_bytes
    total_b = b.total_disk_bytes
    delta = total_b - total_a
    lines.append("  " + "-" * 64)
    lines.append(
        f"  {'TOTAL':18s}  {fmt_bytes(total_a):>14s}  {fmt_bytes(total_b):>14s}  "
        f"{('+' if delta >= 0 else '-')}{fmt_bytes(abs(delta)):>13s}"
    )
    return "\n".join(lines)
