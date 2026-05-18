"""Tests for save_reload_check — adapter save->reload tensor preservation.

Regression target: the AGENTS.md "orphan tensors" bug. An adapter trained
under an older transformers had per-layer k_proj/v_proj LoRA tensors;
after upgrade the new modeling code restructured those modules and PEFT
silently dropped 80 tensors at reload time. The in-memory model and the
reloaded model produced different outputs, but no error fired.

These tests pin down the byte-level diff invariants the rest of the
pipeline (save tripwire + GPU smoke script) relies on.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest
import torch

from src.save_reload_check import (
    StateDiff,
    assert_no_diff,
    diff_state,
    load_adapter_state,
)


def _write_safetensors(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    """Minimal safetensors writer, F32 only — mirrors test_finetune_save_tripwire."""
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


# ---------------------------------------------------------------------------
# diff_state
# ---------------------------------------------------------------------------


def _t(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


def test_diff_state_empty_on_identical_dicts() -> None:
    a = {"k0.lora_A.weight": _t(1.0, 2.0), "k1.lora_B.weight": _t(3.0)}
    b = {"k0.lora_A.weight": _t(1.0, 2.0), "k1.lora_B.weight": _t(3.0)}

    diff = diff_state(a, b)

    assert diff.is_empty()
    assert diff.only_in_a == set()
    assert diff.only_in_b == set()
    assert diff.value_mismatched == []


def test_diff_state_detects_missing_keys() -> None:
    """The original AGENTS.md bug: reload silently drops keys present in
    the saved file. Must surface as only_in_a (in-memory had it, disk
    didn't / reload didn't recover it)."""
    a = {
        "lora_q.weight": _t(1.0),
        "lora_k.weight": _t(2.0),  # the would-be orphan tensor
        "lora_v.weight": _t(3.0),  # the would-be orphan tensor
    }
    b = {"lora_q.weight": _t(1.0)}

    diff = diff_state(a, b)

    assert diff.only_in_a == {"lora_k.weight", "lora_v.weight"}
    assert diff.only_in_b == set()
    assert not diff.is_empty()


def test_diff_state_detects_extra_keys() -> None:
    a = {"lora_q.weight": _t(1.0)}
    b = {"lora_q.weight": _t(1.0), "lora_phantom.weight": _t(9.0)}

    diff = diff_state(a, b)

    assert diff.only_in_b == {"lora_phantom.weight"}
    assert diff.only_in_a == set()


def test_diff_state_detects_byte_mismatch() -> None:
    a = {"lora_q.weight": _t(1.0, 2.0)}
    b = {"lora_q.weight": _t(1.0, 2.5)}

    diff = diff_state(a, b)

    assert diff.only_in_a == set()
    assert diff.only_in_b == set()
    assert "lora_q.weight" in {name for name, *_ in diff.value_mismatched}


def test_diff_state_detects_shape_mismatch() -> None:
    a = {"lora_q.weight": _t(1.0, 2.0)}
    b = {"lora_q.weight": _t(1.0)}

    diff = diff_state(a, b)

    assert any(name == "lora_q.weight" for name, *_ in diff.value_mismatched)


def test_diff_state_detects_dtype_mismatch() -> None:
    a = {"lora_q.weight": torch.tensor([1.0], dtype=torch.float32)}
    b = {"lora_q.weight": torch.tensor([1.0], dtype=torch.float16)}

    diff = diff_state(a, b)

    assert any(name == "lora_q.weight" for name, *_ in diff.value_mismatched)


def test_diff_state_byte_tolerance_pass() -> None:
    """Optional rtol/atol path: bf16<->bf16 numerical equality may differ
    by 1 ulp through float32 conversion. Allow tight tolerance opt-in."""
    a = {"lora_q.weight": torch.tensor([1.0, 2.0], dtype=torch.float32)}
    b = {"lora_q.weight": torch.tensor([1.0 + 1e-7, 2.0], dtype=torch.float32)}

    strict = diff_state(a, b, atol=0.0, rtol=0.0)
    loose = diff_state(a, b, atol=1e-6, rtol=1e-6)

    assert not strict.is_empty()
    assert loose.is_empty()


# ---------------------------------------------------------------------------
# assert_no_diff
# ---------------------------------------------------------------------------


def test_assert_no_diff_passes_on_empty() -> None:
    assert_no_diff(StateDiff(set(), set(), []), label="save")


def test_assert_no_diff_raises_with_label_in_message() -> None:
    diff = StateDiff(
        only_in_a={"lora_k.weight"},
        only_in_b=set(),
        value_mismatched=[],
    )

    with pytest.raises(RuntimeError) as exc:
        assert_no_diff(diff, label="save->reload roundtrip")

    msg = str(exc.value)
    assert "save->reload roundtrip" in msg
    assert "lora_k.weight" in msg
    # The error must mention the AGENTS.md class of bug so operators recognise it.
    assert "orphan" in msg.lower() or "missing" in msg.lower() or "dropped" in msg.lower()


def test_assert_no_diff_message_lists_first_few_keys_truncated() -> None:
    """When the diff is huge (e.g. 80 orphan tensors), the message must
    still be readable. Truncate at a sensible cap with a `(...N more)` hint."""
    many = {f"lora_{i}.weight" for i in range(100)}
    diff = StateDiff(only_in_a=many, only_in_b=set(), value_mismatched=[])

    with pytest.raises(RuntimeError) as exc:
        assert_no_diff(diff, label="save")

    msg = str(exc.value)
    assert "lora_0.weight" in msg
    # Don't dump 100 keys; show a count and a sample.
    assert "100" in msg or "more" in msg.lower()


# ---------------------------------------------------------------------------
# load_adapter_state
# ---------------------------------------------------------------------------


def test_load_adapter_state_reads_single_safetensors(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            "base_model.model.language_model.layers.0.q_proj.lora_A.weight": _t(1.0, 2.0),
            "base_model.model.language_model.layers.0.q_proj.lora_B.weight": _t(3.0, 4.0),
        },
    )

    state = load_adapter_state(adapter)

    assert set(state) == {
        "base_model.model.language_model.layers.0.q_proj.lora_A.weight",
        "base_model.model.language_model.layers.0.q_proj.lora_B.weight",
    }
    assert torch.equal(
        state["base_model.model.language_model.layers.0.q_proj.lora_A.weight"],
        _t(1.0, 2.0),
    )


def test_load_adapter_state_reads_sharded_safetensors(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "model-00001-of-00002.safetensors",
        {"layer0.lora_A.weight": _t(1.0)},
    )
    _write_safetensors(
        adapter / "model-00002-of-00002.safetensors",
        {"layer1.lora_A.weight": _t(2.0)},
    )

    state = load_adapter_state(adapter)

    assert set(state) == {"layer0.lora_A.weight", "layer1.lora_A.weight"}


def test_load_adapter_state_raises_when_no_safetensors(tmp_path: Path) -> None:
    adapter = tmp_path / "empty_adapter"
    adapter.mkdir()

    with pytest.raises(FileNotFoundError):
        load_adapter_state(adapter)


# ---------------------------------------------------------------------------
# Integration: roundtrip helper
# ---------------------------------------------------------------------------


def test_roundtrip_against_disk_passes_when_save_matches_memory(tmp_path: Path) -> None:
    """If get_peft_model_state_dict returned exactly what was written,
    diff_state(in_memory, disk) must be empty. This is the invariant the
    in-pipeline tripwire enforces after every save_model()."""
    from src.save_reload_check import diff_in_memory_vs_disk

    in_memory = {
        "base_model.model.layer0.lora_A.weight": _t(1.0, 2.0),
        "base_model.model.layer0.lora_B.weight": _t(3.0, 4.0),
    }
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(adapter / "adapter_model.safetensors", in_memory)

    diff = diff_in_memory_vs_disk(in_memory, adapter)

    assert diff.is_empty()


def test_roundtrip_against_disk_fires_when_orphan_keys_on_disk(tmp_path: Path) -> None:
    """The exact AGENTS.md bug: in-memory model has 35 layers' worth of
    LoRA tensors, save_pretrained wrote them all, but a stale adapter
    file on disk has tensors no longer present in the in-memory model.
    Surfaces as only_in_b (disk has extras)."""
    from src.save_reload_check import diff_in_memory_vs_disk

    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(1.0),
    }
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {
            "base_model.model.layer0.q_proj.lora_A.weight": _t(1.0),
            # k_proj/v_proj LoRA tensors saved but not in current memory
            "base_model.model.layer0.k_proj.lora_A.weight": _t(9.9),
            "base_model.model.layer0.v_proj.lora_A.weight": _t(9.9),
        },
    )

    diff = diff_in_memory_vs_disk(in_memory, adapter)

    assert not diff.is_empty()
    assert {
        "base_model.model.layer0.k_proj.lora_A.weight",
        "base_model.model.layer0.v_proj.lora_A.weight",
    } <= diff.only_in_b


def test_roundtrip_against_disk_fires_when_memory_has_keys_disk_lacks(tmp_path: Path) -> None:
    """The flip side: trainer.save_model silently dropped some trainable
    tensors. In-memory has them, disk doesn't."""
    from src.save_reload_check import diff_in_memory_vs_disk

    in_memory = {
        "base_model.model.layer0.q_proj.lora_A.weight": _t(1.0),
        "base_model.model.layer0.k_proj.lora_A.weight": _t(2.0),
        "base_model.model.layer0.v_proj.lora_A.weight": _t(3.0),
    }
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    _write_safetensors(
        adapter / "adapter_model.safetensors",
        {"base_model.model.layer0.q_proj.lora_A.weight": _t(1.0)},
    )

    diff = diff_in_memory_vs_disk(in_memory, adapter)

    assert {
        "base_model.model.layer0.k_proj.lora_A.weight",
        "base_model.model.layer0.v_proj.lora_A.weight",
    } <= diff.only_in_a
