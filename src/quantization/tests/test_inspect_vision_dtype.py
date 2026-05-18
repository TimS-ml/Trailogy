"""Tests for the ``inspect_vision_dtype`` tripwire script.

The script asserts the invariant "every tensor under ``vision_tower.*``
must be bf16/fp32" and that the tower's total on-disk size is within a
reasonable window. We synthesize tiny safetensors files in tmp_path so
the tests run without CUDA / MLX / torch.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from scripts.inspect import vision_dtype as inspect_vision_dtype


def _write_safetensors(
    path: Path, tensors: dict[str, dict], data_total: int = 0
) -> None:
    body = json.dumps(tensors).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * data_total)


def _entry(start: int, end: int, dtype: str = "BF16", shape=(2, 2)) -> dict:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}


def _make_dir_with_tower(
    tmp_path: Path, vt_dtype: str, vt_total_bytes: int
) -> Path:
    """Create a model dir whose vision_tower.* tensors sum to
    ``vt_total_bytes`` at the given dtype. One language-model tensor is
    added so the file is non-trivial.
    """
    chunk_size = max(vt_total_bytes // 4, 4)  # 4 vt tensors
    header: dict[str, dict] = {}
    offset = 0
    for i in range(4):
        header[f"model.vision_tower.encoder.layers.{i}.weight"] = _entry(
            offset, offset + chunk_size, dtype=vt_dtype
        )
        offset += chunk_size
    # Add one non-vt tensor so the file isn't all-vt.
    header["model.language_model.lm_head.weight"] = _entry(
        offset, offset + 8, dtype="BF16"
    )
    offset += 8
    _write_safetensors(
        tmp_path / "model.safetensors", header, data_total=offset
    )
    return tmp_path


# ---- happy path -----------------------------------------------------------


def test_bf16_tower_within_bounds_passes(tmp_path):
    """A bf16 vision tower of plausible size returns 0."""
    _make_dir_with_tower(tmp_path, vt_dtype="BF16", vt_total_bytes=320_000_000)
    rc = inspect_vision_dtype.main(
        [
            str(tmp_path),
            "--min_bytes",
            "250000000",
            "--max_bytes",
            "400000000",
        ]
    )
    assert rc == 0


def test_f32_tower_passes(tmp_path):
    """Pure fp32 vision tower is also acceptable."""
    _make_dir_with_tower(tmp_path, vt_dtype="F32", vt_total_bytes=320_000_000)
    rc = inspect_vision_dtype.main(
        [
            str(tmp_path),
            "--min_bytes",
            "250000000",
            "--max_bytes",
            "640000000",
        ]
    )
    assert rc == 0


# ---- failure modes --------------------------------------------------------


def test_u8_tower_fails(tmp_path, capsys):
    """A U8 (NF4-packed) vision tower trips the tripwire."""
    _make_dir_with_tower(tmp_path, vt_dtype="U8", vt_total_bytes=80_000_000)
    rc = inspect_vision_dtype.main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "disallowed dtype" in captured.err
    assert "U8" in captured.err


def test_i32_tower_fails(tmp_path, capsys):
    """GPTQ-packed I32 tensors inside vision_tower also fail (shouldn't
    happen with our GPTQ recipe, but the tripwire defends against it)."""
    _make_dir_with_tower(tmp_path, vt_dtype="I32", vt_total_bytes=80_000_000)
    rc = inspect_vision_dtype.main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "I32" in captured.err


def test_missing_vision_tower_fails(tmp_path, capsys):
    """A model dir with no vision_tower.* keys fails — we never want to
    silently ship a vision-stripped checkpoint to an iOS bundle that
    expects multimodal input."""
    header = {
        "model.language_model.lm_head.weight": _entry(0, 16, "BF16"),
    }
    _write_safetensors(tmp_path / "model.safetensors", header, data_total=16)
    rc = inspect_vision_dtype.main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "no vision_tower" in captured.err


def test_tower_too_small_fails(tmp_path, capsys):
    """Bf16 dtype but tower is suspiciously tiny — sign of partial export."""
    _make_dir_with_tower(tmp_path, vt_dtype="BF16", vt_total_bytes=10_000_000)
    rc = inspect_vision_dtype.main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "< min" in captured.err


def test_tower_too_large_fails(tmp_path, capsys):
    """Bf16 dtype but tower is suspiciously huge — sign of accidental fp32
    upcast or double-write."""
    _make_dir_with_tower(tmp_path, vt_dtype="BF16", vt_total_bytes=900_000_000)
    rc = inspect_vision_dtype.main([str(tmp_path)])
    assert rc == 1
    captured = capsys.readouterr()
    assert "> max" in captured.err


def test_not_a_directory_returns_2(tmp_path, capsys):
    bogus = tmp_path / "does-not-exist"
    rc = inspect_vision_dtype.main([str(bogus)])
    assert rc == 2
    captured = capsys.readouterr()
    assert "Not a directory" in captured.err


# ---- CLI bounds override --------------------------------------------------


def test_custom_max_bytes_allows_larger_tower(tmp_path):
    """When the user knows a different base model has a larger tower,
    they can override --max_bytes and the script accepts it."""
    _make_dir_with_tower(tmp_path, vt_dtype="BF16", vt_total_bytes=800_000_000)
    rc = inspect_vision_dtype.main(
        [str(tmp_path), "--max_bytes", "1000000000"]
    )
    assert rc == 0
