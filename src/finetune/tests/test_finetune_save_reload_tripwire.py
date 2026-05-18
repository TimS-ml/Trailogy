"""Tests for the in-pipeline save->disk tripwire in finetune.py.

Distinct from the GPU-only smoke script in scripts/, this tripwire runs
inside real_train after model.save_pretrained(...). It asserts that what
PEFT wrote to disk is byte-equal to what get_peft_model_state_dict
returned at the same moment. Catches HF/PEFT save-side regressions
(orphan tensors silently dropped, stale tensors from previous runs).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict

import pytest
import torch

from src.finetune import _assert_save_matches_in_memory_state


def _write_safetensors(path: Path, tensors: Dict[str, torch.Tensor]) -> None:
    header: dict = {}
    blob = b""
    offset = 0
    for key, tensor in tensors.items():
        flat = tensor.detach().contiguous().to(torch.float32).cpu().numpy().tobytes()
        header[key] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(flat)],
        }
        blob += flat
        offset += len(flat)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    if pad:
        header_bytes += b" " * pad
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_tripwire_passes_when_disk_matches_in_memory(tmp_path: Path) -> None:
    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1, 0.2),
        "base_model.model.layer0.q_proj.lora_B.weight": _t(0.3, 0.4),
    }
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    _write_safetensors(adapter / "adapter_model.safetensors", in_memory)

    # Must not raise.
    _assert_save_matches_in_memory_state(adapter, in_memory)


def test_tripwire_fires_on_orphan_tensors_on_disk(tmp_path: Path) -> None:
    """AGENTS.md class: disk has k_proj / v_proj LoRA tensors no longer
    present in the in-memory PEFT state — e.g. a stale adapter dir from
    a previous run was reused without cleanup."""
    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1),
    }
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1),
            # 80 hypothetical orphans, abbreviated:
            "base_model.model.layer0.k_proj.lora_A.weight": _t(9.9),
            "base_model.model.layer0.v_proj.lora_A.weight": _t(9.9),
        },
    )

    with pytest.raises(RuntimeError, match=r"orphan"):
        _assert_save_matches_in_memory_state(adapter, in_memory)


def test_tripwire_fires_when_save_silently_dropped_tensors(tmp_path: Path) -> None:
    """Flip side: trainer.save_model wrote fewer tensors than the in-memory
    PEFT model had. Indicates a save-path regression."""
    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1),
        "base_model.model.layer0.k_proj.lora_A.weight": _t(0.2),
        "base_model.model.layer0.v_proj.lora_A.weight": _t(0.3),
    }
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {"base_model.model.layer0.q_proj.lora_A.weight": _t(0.1)},
    )

    with pytest.raises(RuntimeError, match=r"orphan"):
        _assert_save_matches_in_memory_state(adapter, in_memory)


def test_tripwire_fires_on_byte_corruption(tmp_path: Path) -> None:
    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1, 0.2),
    }
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {"base_model.model.layer0.q_proj.lora_A.weight": _t(0.1, 0.25)},
    )

    with pytest.raises(RuntimeError, match=r"mismatch"):
        _assert_save_matches_in_memory_state(adapter, in_memory)


def test_tripwire_message_names_agents_md_bug_class(tmp_path: Path) -> None:
    """Operator-facing error must point at AGENTS.md so the meaning is
    immediately recognizable in a stack trace."""
    in_memory = {"base_model.model.layer0.q_proj.lora_A.weight": _t(0.1)}
    adapter = tmp_path / "final-adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            "base_model.model.layer0.q_proj.lora_A.weight": _t(0.1),
            "base_model.model.layer0.k_proj.lora_A.weight": _t(9.9),
        },
    )

    with pytest.raises(RuntimeError) as exc:
        _assert_save_matches_in_memory_state(adapter, in_memory)

    assert "AGENTS.md" in str(exc.value)
