"""Tests for export_mlx._assert_projector_changed_if_tuned.

This tripwire fires at export time when the LoRA adapter directory
contains projector tensors (= ``modules_to_save`` was used during
training) but those tensors in the merged model are bit-identical to
the base model. That combination means PEFT silently failed to restore
the modules_to_save weights — the export would otherwise ship a model
that's no better than a LoRA-only run, with no warning.

LoRA-only adapters (the existing flow) do not contain projector
tensors, so the check is a no-op for them. This is the backward-compat
guarantee: every existing export keeps working unchanged.

Tests synthesize tiny safetensors files by hand (no torch / mlx
required), same approach as test_export_mlx.py.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

from src.export_mlx import _assert_projector_changed_if_tuned


# ---------------------------------------------------------------------------
# Fakes mirroring test_freeze.py / test_projector.py
# ---------------------------------------------------------------------------


@dataclass
class FakeParam:
    name: str
    data: bytes  # contents we'll diff for "changed?" check
    requires_grad: bool = True


class FakeModel:
    def __init__(self, params: list[tuple[str, FakeParam]]) -> None:
        self.params = params

    def named_parameters(self) -> Iterator[tuple[str, FakeParam]]:
        for n, p in self.params:
            yield n, p


# ---------------------------------------------------------------------------
# Tiny safetensors writer with real data bytes (so the tripwire can diff).
# ---------------------------------------------------------------------------


def _write_safetensors(
    path: Path,
    tensors: dict[str, bytes],
) -> None:
    """Write a syntactically valid safetensors with real F32-shaped data
    bytes per tensor. The tripwire reads tensor data via safetensors,
    not headers — so we need a real data section."""
    header: dict = {}
    blob = b""
    offset = 0
    for key, data in tensors.items():
        # Pretend each tensor is a 1D F32 array. shape = [len(data)//4].
        n_elems = len(data) // 4
        if n_elems * 4 != len(data):
            raise ValueError("tensor data length must be a multiple of 4 (F32)")
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
# No-op for LoRA-only adapters
# ---------------------------------------------------------------------------


def test_tripwire_inactive_for_lora_only_adapter(tmp_path: Path) -> None:
    """If the adapter dir has only lora_* tensors (no projector tensors),
    the tripwire skips the check and does not raise — even if base and
    merged are identical."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1, 0.2),
            "base_model.model.language_model.layers.0.q_proj.lora_B.weight": _f32(0.3, 0.4),
        },
    )
    base = FakeModel([("embed_vision.embedding_projection.weight",
                      FakeParam("ev", data=_f32(1.0, 2.0)))])
    merged = FakeModel([("embed_vision.embedding_projection.weight",
                        FakeParam("ev", data=_f32(1.0, 2.0)))])  # identical
    # Should not raise — no projector tensors in adapter, check is a no-op.
    _assert_projector_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_inactive_when_no_adapter_dir(tmp_path: Path) -> None:
    """A merge driven from --merged_dir (no adapter_path) → adapter_dir
    is None or doesn't exist. Don't crash."""
    base = FakeModel([("embed_vision.embedding_projection.weight",
                      FakeParam("ev", data=_f32(1.0)))])
    merged = FakeModel([("embed_vision.embedding_projection.weight",
                        FakeParam("ev", data=_f32(2.0)))])
    _assert_projector_changed_if_tuned(base, merged, None)
    # And on a non-existent path:
    _assert_projector_changed_if_tuned(base, merged, tmp_path / "no_adapter")


# ---------------------------------------------------------------------------
# Active checks
# ---------------------------------------------------------------------------


def test_tripwire_passes_when_projector_changed(tmp_path: Path) -> None:
    """Adapter has projector tensors, merged != base → check passes."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            # PEFT's modules_to_save naming: <module_name>.modules_to_save.default.<param>
            # but the matcher works on substring tokens, so any name containing
            # `embed_vision.` qualifies.
            "base_model.model.embed_vision.modules_to_save.default.embedding_projection.weight":
                _f32(0.5, 0.6, 0.7, 0.8),
            "base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1, 0.2),
        },
    )
    base = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(1.0, 2.0))),
    ])
    merged = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(9.9, 8.8))),
    ])
    _assert_projector_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_fires_when_projector_unchanged(tmp_path: Path) -> None:
    """Adapter has projector tensors, but merged == base → ship-stopper.

    This is the silent-PEFT-regression scenario: training succeeded, the
    adapter was saved, but at load+merge time PEFT failed to restore the
    full-param projector weights. Without the tripwire we'd ship a model
    that performs no better than LoRA-only.
    """
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {
            "base_model.model.embed_vision.modules_to_save.default.embedding_projection.weight":
                _f32(0.5, 0.6),
        },
    )
    base = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(1.0, 2.0))),
    ])
    merged = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(1.0, 2.0))),  # same
    ])
    with pytest.raises(RuntimeError, match="projector"):
        _assert_projector_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_handles_sharded_adapter(tmp_path: Path) -> None:
    """Projector tensor lives in shard 2 of 2."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "model-00001-of-00002.safetensors",
        {"base_model.model.language_model.layers.0.q_proj.lora_A.weight": _f32(0.1, 0.2)},
    )
    _write_safetensors(
        adapter_dir / "model-00002-of-00002.safetensors",
        {"base_model.model.embed_vision.embedding_projection.weight": _f32(0.5, 0.6)},
    )
    base = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(1.0, 2.0))),
    ])
    merged = FakeModel([
        ("embed_vision.embedding_projection.weight", FakeParam("ev", data=_f32(1.0, 2.0))),
    ])
    with pytest.raises(RuntimeError, match="projector"):
        _assert_projector_changed_if_tuned(base, merged, adapter_dir)


def test_tripwire_skips_check_when_merged_lacks_projector(tmp_path: Path) -> None:
    """Defensive: if for some reason the merged model has no projector
    params we can compare against, log + skip rather than crash. The
    `_model_has_vision_tower` tripwire already catches the more
    catastrophic case of no vision params at all."""
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    _write_safetensors(
        adapter_dir / "adapter_model.safetensors",
        {"base_model.model.embed_vision.embedding_projection.weight": _f32(0.5, 0.6)},
    )
    # Both base and merged have NO embed_vision params (pathological).
    base = FakeModel([
        ("language_model.layers.0.q_proj.weight", FakeParam("lq", data=_f32(1.0))),
    ])
    merged = FakeModel([
        ("language_model.layers.0.q_proj.weight", FakeParam("lq", data=_f32(2.0))),
    ])
    # Should not raise — there's nothing to compare against, skip with a warning.
    _assert_projector_changed_if_tuned(base, merged, adapter_dir)
