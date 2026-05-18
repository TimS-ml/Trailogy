"""Safetensors header inspection — no tensor data load.

Reading just the header lets us enumerate keys, dtypes, shapes, and
on-disk byte ranges without paying the memory cost of materializing
weights. This is the primary tool for comparing two quantized
checkpoints (e.g. mlx-community 3.58 GB vs our reproduction).

The on-disk layout (per safetensors spec):

    | u64 little-endian: header_len = N
    | N bytes of JSON: { "tensor_name": {"dtype": str, "shape": [...],
    |                                    "data_offsets": [start, end]},
    |                     ... ,
    |                     "__metadata__": {...} (optional) }
    | tensor data, indexed by data_offsets

This module mirrors the header-reader pattern in
``finetune/src/export_mlx.py`` but exposes it as a reusable helper.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path


HEADER_LEN_BYTES = 8


@dataclass(frozen=True)
class TensorEntry:
    """One entry from a safetensors header.

    ``nbytes`` is the on-disk size of the tensor's data segment, i.e.
    ``data_offsets[1] - data_offsets[0]``. This is the size that actually
    contributes to file-on-disk size — independent of dtype, useful for
    measuring "how many bytes does this tensor cost".
    """

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def bits_per_element(self) -> float:
        """Effective bits/element on disk. Useful for spotting non-uniform
        quantization (e.g. some tensors at 4 bits, others at 16 bits).
        """
        if self.numel == 0:
            return 0.0
        return self.nbytes * 8.0 / self.numel


def read_header(path: Path) -> tuple[int, dict]:
    """Read a single safetensors file's header. Returns
    ``(data_segment_offset, header_dict)``.

    ``header_dict`` includes ``__metadata__`` if present.
    """
    path = Path(path)
    with path.open("rb") as fin:
        raw = fin.read(HEADER_LEN_BYTES)
        if len(raw) != HEADER_LEN_BYTES:
            raise ValueError(f"{path}: file too short to be safetensors")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 0 or header_len > 64 * 1024 * 1024:
            raise ValueError(
                f"{path}: implausible header length {header_len}"
            )
        body = fin.read(header_len)
        if len(body) != header_len:
            raise ValueError(f"{path}: truncated header")
    header = json.loads(body)
    return HEADER_LEN_BYTES + header_len, header


def enumerate_tensors(path: Path) -> list[TensorEntry]:
    """Return all tensor entries (excluding ``__metadata__``) from one
    safetensors file.
    """
    _, header = read_header(path)
    entries: list[TensorEntry] = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(meta, dict):
            continue
        try:
            offsets = meta["data_offsets"]
            entries.append(
                TensorEntry(
                    name=name,
                    dtype=str(meta["dtype"]),
                    shape=tuple(int(d) for d in meta.get("shape", [])),
                    data_offsets=(int(offsets[0]), int(offsets[1])),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{path}: malformed entry for {name!r}: {e}"
            ) from e
    return entries


def enumerate_directory(directory: Path) -> list[TensorEntry]:
    """Enumerate every safetensors shard in a directory and return a
    flat list of all tensor entries. Sharded checkpoints (``model-00001-of-00003.safetensors``)
    are handled transparently.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    entries: list[TensorEntry] = []
    for st in sorted(directory.glob("*.safetensors")):
        entries.extend(enumerate_tensors(st))
    return entries


def bucket_by_submodule(
    entries: list[TensorEntry],
    buckets: tuple[str, ...] = (
        "language_model",
        "vision_tower",
        "audio_tower",
        "embed_vision",
        "embed_audio",
    ),
    other_label: str = "other",
) -> dict[str, list[TensorEntry]]:
    """Group entries by which submodule prefix their name matches.

    The matcher is the same substring-with-trailing-dot logic used in
    ``finetune/src/freeze.py`` to be robust to PEFT/HF wrapping
    prefixes (``base_model.model....``).
    """
    grouped: dict[str, list[TensorEntry]] = {b: [] for b in buckets}
    grouped[other_label] = []
    for e in entries:
        placed = False
        for b in buckets:
            if f"{b}." in e.name:
                grouped[b].append(e)
                placed = True
                break
        if not placed:
            grouped[other_label].append(e)
    return grouped


def summarize_size(entries: list[TensorEntry]) -> dict:
    """Aggregate stats over a list of tensor entries.

    Returns:
        {
            "total_bytes": int,
            "total_numel": int,
            "avg_bits_per_element": float,
            "dtype_histogram": {dtype: (n_tensors, total_bytes, total_numel)},
        }
    """
    total_bytes = sum(e.nbytes for e in entries)
    total_numel = sum(e.numel for e in entries)
    dtype_hist: dict[str, list[int]] = {}
    for e in entries:
        slot = dtype_hist.setdefault(e.dtype, [0, 0, 0])
        slot[0] += 1
        slot[1] += e.nbytes
        slot[2] += e.numel
    return {
        "total_bytes": total_bytes,
        "total_numel": total_numel,
        "avg_bits_per_element": (
            total_bytes * 8.0 / total_numel if total_numel else 0.0
        ),
        "dtype_histogram": {
            dt: tuple(vals) for dt, vals in dtype_hist.items()
        },
    }
