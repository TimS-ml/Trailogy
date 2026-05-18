"""Tests for export_mlx._assert_vision_layers_changed_if_tuned.

Parallel to test_export_projector_tripwire.py. Fires when the LoRA
adapter directory contains vision-encoder-layer tensors (=
``tune_last_n_vision_layers > 0`` and PEFT serialized the
``modules_to_save`` wrapper for those layers) but those tensors in the
merged model are byte-identical to the base — meaning PEFT silently
dropped the modules_to_save weights and the export is wasted.

LoRA-only adapters and projector-only adapters do NOT contain
vision-encoder-layer tensors, so the check is a no-op for them.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from src.export_mlx import (
    _adapter_has_vision_layer_tensors,
    _assert_vision_layers_changed_if_tuned,
    _model_param_data_bytes_by_vision_layer,
)


# ---------------------------------------------------------------------------
# Fakes (mirroring test_export_projector_tripwire.py)
# ---------------------------------------------------------------------------


@dataclass
class FakeParam:
    name: str
    data: bytes
    requires_grad: bool = True


class FakeModel:
    def __init__(self, params: list[tuple[str, FakeParam]]) -> None:
        self.params = params

    def named_parameters(self) -> Iterator[tuple[str, FakeParam]]:
        for n, p in self.params:
            yield n, p


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header: dict = {}
    blob = b""
    offset = 0
    for key, data in tensors.items():
        n_elems = len(data) // 4
        if n_elems * 4 != len(data):
            raise ValueError("F32 data length must be multiple of 4")
        header[key] = {
            "dtype": "F32",
            "shape": [n_elems],
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


def _f32(*vals: float) -> bytes:
    return struct.pack(f"<{len(vals)}f", *vals)


# ---------------------------------------------------------------------------
# _adapter_has_vision_layer_tensors
# ---------------------------------------------------------------------------


def test_detect_returns_empty_for_lora_only(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {"base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1)},
    )
    assert _adapter_has_vision_layer_tensors(adapter_dir) == {}


def test_detect_returns_empty_for_projector_only_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.embed_vision.modules_to_save.default."
            "embedding_projection.weight": _f32(0.5, 0.6),
        },
    )
    assert _adapter_has_vision_layer_tensors(adapter_dir) == {}


def test_detect_finds_vision_layer_tensors_grouped_by_index(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.1, 0.2),
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.mlp.fc1.weight": _f32(0.3, 0.4),
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5, 0.6),
        },
    )
    by_idx = _adapter_has_vision_layer_tensors(adapter_dir)
    assert set(by_idx.keys()) == {14, 15}
    assert len(by_idx[14]) == 2
    assert len(by_idx[15]) == 1


def test_detect_excludes_original_module_copies(tmp_path: Path) -> None:
    """PEFT's frozen reference copy must not count — only the
    .modules_to_save.{adapter}. copies are "tuned"."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.15"
            ".original_module.attn.q_proj.weight": _f32(0.1),
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.2),
        },
    )
    by_idx = _adapter_has_vision_layer_tensors(adapter_dir)
    assert set(by_idx.keys()) == {15}
    # Only the modules_to_save key should be in the list.
    assert all(".modules_to_save." in k for k in by_idx[15])
    assert all(".original_module." not in k for k in by_idx[15])


def test_detect_handles_missing_adapter_dir(tmp_path: Path) -> None:
    assert _adapter_has_vision_layer_tensors(None) == {}
    assert _adapter_has_vision_layer_tensors(tmp_path / "nope") == {}


def test_detect_handles_sharded_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "model-00001-of-00002.safetensors",
        {
            "base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1),
        },
    )
    _write_safetensors(
        adapter_dir / "model-00002-of-00002.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5),
        },
    )
    by_idx = _adapter_has_vision_layer_tensors(adapter_dir)
    assert 15 in by_idx


def test_detect_rejects_lookalike_paths(tmp_path: Path) -> None:
    """Look-alikes like vision_tower_descriptor must not match."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower_descriptor.encoder.layers.14.weight": _f32(0.1),
            "base_model.model.language_model.layers.14.q_proj.weight": _f32(0.2),
        },
    )
    assert _adapter_has_vision_layer_tensors(adapter_dir) == {}


# ---------------------------------------------------------------------------
# _assert_vision_layers_changed_if_tuned
# ---------------------------------------------------------------------------


def test_tripwire_inactive_for_lora_only_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {"base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1)},
    )
    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0))),  # identical — OK because no adapter tensors
    ])
    _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_inactive_for_projector_only_adapter(tmp_path: Path) -> None:
    """Adapter has projector tensors but no vision-layer tensors → skip."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.embed_vision.modules_to_save.default."
            "embedding_projection.weight": _f32(0.5, 0.6),
        },
    )
    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0))),
    ])
    _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_inactive_when_no_adapter_dir() -> None:
    base = FakeModel([])
    merged = FakeModel([])
    _assert_vision_layers_changed_if_tuned(base, merged, None)


def test_tripwire_passes_when_vision_layers_changed(tmp_path: Path) -> None:
    """Adapter has vision-layer tensors AND merged != base → check passes."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5),
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.6),
        },
    )
    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0, 2.0))),
        ("vision_tower.encoder.layers.15.attn.q_proj.weight",
         FakeParam("v15", data=_f32(3.0, 4.0))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(7.0, 8.0))),
        ("vision_tower.encoder.layers.15.attn.q_proj.weight",
         FakeParam("v15", data=_f32(9.0, 0.5))),
    ])
    _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_fires_when_vision_layers_unchanged(tmp_path: Path) -> None:
    """Adapter contains vision-layer tensors but the merged params are
    bit-identical to base → ship-stopper."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5),
        },
    )
    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0, 2.0))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0, 2.0))),  # same
    ])
    with pytest.raises(RuntimeError, match="vision"):
        _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_fires_when_only_one_tuned_layer_unchanged(tmp_path: Path) -> None:
    """If layer 14 changed but layer 15 (also in adapter) is byte-identical
    in merged, fire — the export is partially broken."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5),
            "base_model.model.vision_tower.encoder.layers.15"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.6),
        },
    )
    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(1.0))),
        ("vision_tower.encoder.layers.15.attn.q_proj.weight",
         FakeParam("v15", data=_f32(2.0))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=_f32(7.0))),    # changed
        ("vision_tower.encoder.layers.15.attn.q_proj.weight",
         FakeParam("v15", data=_f32(2.0))),    # UNCHANGED — should trigger
    ])
    with pytest.raises(RuntimeError, match="vision"):
        _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_fires_for_unchanged_real_bfloat16_tensors(tmp_path: Path) -> None:
    """Real export loads the HF merge base as bf16. PyTorch bf16 tensors
    cannot be converted with tensor.numpy() directly; the tripwire must
    still collect bytes and fire when merged == base."""
    torch = pytest.importorskip("torch")

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.vision_tower.encoder.layers.14"
            ".modules_to_save.default.attn.q_proj.weight": _f32(0.5),
        },
    )

    base = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=torch.tensor([1.0, 2.0], dtype=torch.bfloat16))),
    ])
    merged = FakeModel([
        ("vision_tower.encoder.layers.14.attn.q_proj.weight",
         FakeParam("v14", data=torch.tensor([1.0, 2.0], dtype=torch.bfloat16))),
    ])

    # The snapshot helper must not silently drop bf16 tensors.
    snapshot = _model_param_data_bytes_by_vision_layer(base, [14])
    assert 14 in snapshot
    assert "vision_tower.encoder.layers.14.attn.q_proj.weight" in snapshot[14]

    with pytest.raises(RuntimeError, match="vision"):
        _assert_vision_layers_changed_if_tuned(base, merged, adapter_dir)
