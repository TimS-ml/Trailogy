"""Tests for finetune.py save-side modules_to_save tripwires."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.finetune import _assert_vision_layer_tensors_present_if_tuned


def _write_safetensors(path: Path, keys: list[str]) -> None:
    header: dict = {}
    blob = b""
    offset = 0
    for key in keys:
        data = struct.pack("<f", 1.0)
        header[key] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + len(data)],
        }
        blob += data
        offset += len(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad
    with path.open("wb") as fout:
        fout.write(struct.pack("<Q", len(header_bytes)))
        fout.write(header_bytes)
        fout.write(blob)


def test_vision_layer_save_tripwire_no_ops_when_not_tuned(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        ["base_model.model.language_model.layers.0.q_proj.lora_A.weight"],
    )

    _assert_vision_layer_tensors_present_if_tuned(adapter_dir, [])


def test_vision_layer_save_tripwire_passes_when_all_indices_present(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        [
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight",
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.mlp.fc1.weight",
        ],
    )

    _assert_vision_layer_tensors_present_if_tuned(adapter_dir, [14, 15])


def test_vision_layer_save_tripwire_fires_when_index_missing(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        [
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight",
        ],
    )

    with pytest.raises(RuntimeError, match="missing tensors.*15"):
        _assert_vision_layer_tensors_present_if_tuned(adapter_dir, [14, 15])


def test_vision_layer_save_tripwire_ignores_original_module_copy(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        [
            "base_model.model.vision_tower.encoder.layers.15"
            ".original_module.attn.q_proj.weight",
        ],
    )

    with pytest.raises(RuntimeError, match="missing tensors.*15"):
        _assert_vision_layer_tensors_present_if_tuned(adapter_dir, [15])


def test_vision_layer_save_tripwire_handles_sharded_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "model-00001-of-00002.safetensors",
        [
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight",
        ],
    )
    _write_safetensors(
        adapter_dir / "model-00002-of-00002.safetensors",
        [
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight",
        ],
    )

    _assert_vision_layer_tensors_present_if_tuned(adapter_dir, [14, 15])
