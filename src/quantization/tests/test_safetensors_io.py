"""Pure-python tests for ``src.common.safetensors_io``.

Synthesizes minimal safetensors files in tmp_path so the tests run
on Mac without CUDA / MLX / torch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.common.safetensors_io import (
    bucket_by_submodule,
    enumerate_directory,
    enumerate_tensors,
    read_header,
    summarize_size,
)


def _write_safetensors(
    path: Path, tensors: dict[str, dict], data_total: int = 0
) -> None:
    """Write a minimal safetensors file with the given header + ``data_total``
    bytes of zero data. ``tensors`` is the header dict (already containing
    ``data_offsets``).
    """
    body = json.dumps(tensors).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * data_total)


def _make_entry(start: int, end: int, dtype: str = "BF16", shape=(2, 2)) -> dict:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}


def test_read_header_round_trip(tmp_path):
    path = tmp_path / "tiny.safetensors"
    header = {
        "language_model.foo": _make_entry(0, 16, "BF16", (2, 4)),
        "vision_tower.bar": _make_entry(16, 24, "F16", (2, 2)),
        "__metadata__": {"format": "pt"},
    }
    _write_safetensors(path, header, data_total=24)
    data_off, h = read_header(path)
    assert data_off > 8
    assert "language_model.foo" in h
    assert "__metadata__" in h


def test_enumerate_tensors_skips_metadata(tmp_path):
    path = tmp_path / "tiny.safetensors"
    header = {
        "a.weight": _make_entry(0, 16, "BF16", (2, 4)),
        "__metadata__": {"format": "pt"},
    }
    _write_safetensors(path, header, data_total=16)
    entries = enumerate_tensors(path)
    assert len(entries) == 1
    e = entries[0]
    assert e.name == "a.weight"
    assert e.nbytes == 16
    assert e.numel == 8
    assert e.bits_per_element == 16.0  # 16 bytes / 8 elts * 8 bits = 16 bits/elt


def test_bucket_by_submodule_substring_match(tmp_path):
    path = tmp_path / "tiny.safetensors"
    header = {
        # Realistic PEFT-wrapped name — must still bucket correctly:
        "base_model.model.language_model.x.weight": _make_entry(0, 8),
        "model.vision_tower.encoder.layers.0.weight": _make_entry(8, 16),
        "audio_tower.layers.0.weight": _make_entry(16, 24),
        "embed_vision.proj.weight": _make_entry(24, 32),
        "embed_audio.proj.weight": _make_entry(32, 40),
        # Look-alike that should NOT match vision_tower:
        "vision_tower_descriptor.tag": _make_entry(40, 48),
    }
    _write_safetensors(path, header, data_total=48)
    entries = enumerate_tensors(path)
    grouped = bucket_by_submodule(entries)
    assert len(grouped["language_model"]) == 1
    assert len(grouped["vision_tower"]) == 1
    assert len(grouped["audio_tower"]) == 1
    assert len(grouped["embed_vision"]) == 1
    assert len(grouped["embed_audio"]) == 1
    # `vision_tower_descriptor.tag` doesn't contain "vision_tower." with
    # trailing dot, so it should land in "other".
    assert len(grouped["other"]) == 1
    assert grouped["other"][0].name == "vision_tower_descriptor.tag"


def test_summarize_size_aggregates(tmp_path):
    path = tmp_path / "tiny.safetensors"
    header = {
        "a.weight": _make_entry(0, 16, "BF16", (2, 4)),
        "b.weight": _make_entry(16, 24, "F16", (2, 2)),
        "c.weight": _make_entry(24, 28, "U8", (2, 2)),  # 4-bit-ish: 2 bits/elt
    }
    _write_safetensors(path, header, data_total=28)
    entries = enumerate_tensors(path)
    summary = summarize_size(entries)
    assert summary["total_bytes"] == 28
    assert summary["total_numel"] == 16  # 8 + 4 + 4
    # 28 bytes * 8 bits / 16 elts = 14 bits/elt
    assert summary["avg_bits_per_element"] == pytest.approx(14.0)
    assert set(summary["dtype_histogram"]) == {"BF16", "F16", "U8"}


def test_enumerate_directory_multiple_shards(tmp_path):
    shard_a = tmp_path / "model-00001-of-00002.safetensors"
    shard_b = tmp_path / "model-00002-of-00002.safetensors"
    _write_safetensors(
        shard_a,
        {"x.weight": _make_entry(0, 16, "BF16", (2, 4))},
        data_total=16,
    )
    _write_safetensors(
        shard_b,
        {"y.weight": _make_entry(0, 16, "F16", (2, 4))},
        data_total=16,
    )
    entries = enumerate_directory(tmp_path)
    names = {e.name for e in entries}
    assert names == {"x.weight", "y.weight"}


def test_truncated_header_raises(tmp_path):
    path = tmp_path / "broken.safetensors"
    path.write_bytes(b"\x00\x00\x00")  # less than 8 bytes
    with pytest.raises(ValueError, match="too short"):
        read_header(path)


def test_implausible_header_length_raises(tmp_path):
    path = tmp_path / "wonky.safetensors"
    # claim header is 1 GB long
    path.write_bytes(struct.pack("<Q", 1 << 32))
    with pytest.raises(ValueError, match="implausible header"):
        read_header(path)
