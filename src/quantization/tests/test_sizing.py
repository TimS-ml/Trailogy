"""Pure-python tests for the directory sizing helper."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.common.sizing import (
    diff_directories,
    fmt_bytes,
    measure_directory,
)


def _write_safetensors(path: Path, tensors: dict[str, dict], data_total: int) -> None:
    body = json.dumps(tensors).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(body)))
        f.write(body)
        f.write(b"\x00" * data_total)


def _entry(start: int, end: int, dtype: str = "BF16", shape=(2, 2)) -> dict:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": [start, end]}


def test_fmt_bytes_thresholds():
    assert fmt_bytes(0) == "0 B"
    assert fmt_bytes(1023) == "1023 B"
    assert fmt_bytes(1024).endswith("KB")
    assert fmt_bytes(2 << 20).endswith("MB")
    assert fmt_bytes(3 << 30).endswith("GB")


def test_measure_directory_per_submodule(tmp_path):
    p = tmp_path / "model.safetensors"
    _write_safetensors(
        p,
        {
            "language_model.x": _entry(0, 16, "BF16", (2, 4)),
            "vision_tower.y": _entry(16, 24, "F16", (2, 2)),
            "embed_vision.z": _entry(24, 28, "U8", (2, 2)),
        },
        data_total=28,
    )
    # Add some non-safetensors files so non_safetensors_bytes is exercised.
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text(json.dumps({"foo": "bar"}))

    rep = measure_directory(tmp_path)
    assert rep.per_submodule_bytes["language_model"] == 16
    assert rep.per_submodule_bytes["vision_tower"] == 8
    assert rep.per_submodule_bytes["embed_vision"] == 4
    assert rep.per_submodule_bytes["audio_tower"] == 0
    assert rep.total_safetensors_bytes == 28
    assert rep.non_safetensors_bytes > 0
    text = rep.format_report()
    assert "language_model" in text
    assert "vision_tower" in text


def test_diff_directories_signed_delta(tmp_path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    a_dir.mkdir()
    b_dir.mkdir()
    _write_safetensors(
        a_dir / "model.safetensors",
        {"language_model.x": _entry(0, 16, "BF16", (2, 4))},
        data_total=16,
    )
    _write_safetensors(
        b_dir / "model.safetensors",
        {"language_model.x": _entry(0, 8, "U8", (2, 4))},
        data_total=8,
    )
    a = measure_directory(a_dir)
    b = measure_directory(b_dir)
    text = diff_directories(a, b, label_a="big", label_b="small")
    # B is smaller, so delta should be negative.
    assert "big" in text and "small" in text
    assert "-" in text  # signed delta line
