"""Tests for the GPTQ calibration data builder.

The actual quantize() call needs a 9.5 GB bf16 model + gptqmodel +
CUDA; that's a smoke test, not a unit test. We test the pure-python
calibration assembly here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.methods.gptq import (
    CalibrationDataLeakError,
    GPTQConfig,
    _load_plantnet_calibration,
    _reject_calibration_leak,
    build_calibration_dataset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_plantnet_calibration_handles_string_content(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "image": "/tmp/p1.jpg",
                "conversations": [
                    {"role": "user", "content": "What plant is this?"},
                    {"role": "assistant", "content": "This is Quercus robur."},
                ],
            },
            {
                "image": "/tmp/p2.jpg",
                "conversations": [
                    {"role": "user", "content": "And this one?"},
                    {"role": "assistant", "content": "This is Acer rubrum."},
                ],
            },
        ],
    )
    calib = _load_plantnet_calibration(path, n_samples=10, seed=0)
    assert len(calib) == 2
    assert all("text" in r for r in calib)
    assert any("Quercus robur" in r["text"] for r in calib)


def test_plantnet_calibration_handles_block_content(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "image": "/tmp/p1.jpg",
                "conversations": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": "What plant is this?"},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "This is X."}],
                    },
                ],
            }
        ],
    )
    calib = _load_plantnet_calibration(path, n_samples=10, seed=0)
    assert len(calib) == 1
    assert "What plant is this?" in calib[0]["text"]
    assert "This is X." in calib[0]["text"]


def test_plantnet_calibration_respects_n_samples(tmp_path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "image": f"/tmp/p{i}.jpg",
                "conversations": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
            }
            for i in range(20)
        ],
    )
    calib = _load_plantnet_calibration(path, n_samples=5, seed=0)
    assert len(calib) == 5


def test_plantnet_calibration_missing_file_returns_empty(tmp_path):
    calib = _load_plantnet_calibration(tmp_path / "does-not-exist.jsonl", 100, seed=0)
    assert calib == []


def test_build_calibration_with_zero_text_avoids_dataset_download(tmp_path):
    """Verify n_calib_text=0 path is short-circuited so the test doesn't
    actually try to download wikitext."""
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {
                "image": "/tmp/p1.jpg",
                "conversations": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                ],
            }
        ],
    )
    cfg = GPTQConfig(n_calib_plantnet=1, n_calib_text=0)
    calib = build_calibration_dataset(cfg, plantnet_jsonl=path)
    assert len(calib) == 1


# ---------------------------------------------------------------------------
# Data-leak guards — these protect against accidentally feeding eval data
# (val.jsonl) or overfit100 data (train==eval) into GPTQ calibration.
# ---------------------------------------------------------------------------


def test_reject_calibration_leak_rejects_val_jsonl(tmp_path):
    p = tmp_path / "val.jsonl"
    p.touch()
    with pytest.raises(CalibrationDataLeakError, match="val.jsonl"):
        _reject_calibration_leak(p)


def test_reject_calibration_leak_rejects_nested_val_jsonl(tmp_path):
    nested = tmp_path / "data" / "val.jsonl"
    nested.parent.mkdir()
    nested.touch()
    with pytest.raises(CalibrationDataLeakError):
        _reject_calibration_leak(nested)


def test_reject_calibration_leak_rejects_overfit100(tmp_path):
    p = tmp_path / "plantnet-overfit100-train.jsonl"
    p.touch()
    with pytest.raises(CalibrationDataLeakError, match="overfit100"):
        _reject_calibration_leak(p)


def test_reject_calibration_leak_accepts_train_jsonl(tmp_path):
    p = tmp_path / "train.jsonl"
    p.touch()
    # Should not raise.
    _reject_calibration_leak(p)


def test_load_plantnet_calibration_fails_loudly_on_val(tmp_path):
    """End-to-end check that loading val.jsonl as calibration raises."""
    p = tmp_path / "val.jsonl"
    _write_jsonl(p, [
        {
            "image": "/tmp/p.jpg",
            "conversations": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
        }
    ])
    with pytest.raises(CalibrationDataLeakError):
        _load_plantnet_calibration(p, n_samples=1, seed=0)
