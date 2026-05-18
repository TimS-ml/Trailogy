"""Regression tests for ``run_quant.py``'s merge path.

These tests pin down two contracts:

1. **When ``--adapter`` is given**, ``run_quant.main`` MUST merge the
   adapter via ``scripts.repair.merge_safetensors.merge`` — the
   safetensors-level tool that preserves the full base key set.
   It MUST NOT go through ``common.model_io.load_bf16_multimodal``
   (which uses HF transformers' ``save_pretrained`` and silently drops
   ``k_proj`` / ``v_proj`` / ``k_norm`` / ``v_norm`` from KV-shared
   layers — Gemma 4 E2B layers 15-34 — because those tensors aren't
   registered ``nn.Parameter``s in the Gemma 4 class).

   The HF path's drop is documented in:
   - ``scripts/merge_safetensors.py`` module docstring
   - ``AGENTS.md`` known-bug list (transformers v5.8)

2. **When ``--adapter`` is omitted**, ``run_quant.main`` should NOT
   re-save the base via HF ``save_pretrained`` either — just resolve
   the base dir and hand it to the quant dispatcher untouched.

Synthetic safetensors are used so the test runs in <100 ms on a
laptop with no CUDA, no MLX, no HF network access.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _write_fake_base(base_dir: Path, n_layers: int = 3) -> dict[str, "torch.Tensor"]:
    """Write a fake bf16 base with multiple layers' q_proj + k_proj.

    Returns the in-memory tensor dict so callers can compare.
    """
    import torch
    from safetensors.torch import save_file

    base_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        layer = f"model.language_model.layers.{i}.self_attn"
        tensors[f"{layer}.q_proj.weight"] = torch.zeros(4, 4, dtype=torch.bfloat16)
        # k_proj kept narrow — this is the tensor the HF save_pretrained path
        # silently drops on Gemma 4 E2B KV-shared layers.
        tensors[f"{layer}.k_proj.weight"] = torch.zeros(2, 4, dtype=torch.bfloat16)
    save_file(tensors, str(base_dir / "model.safetensors"))

    # Minimal side-cars — merge_safetensors._copy_sidecars looks for these.
    (base_dir / "config.json").write_text(json.dumps({"architectures": ["Gemma4ForConditionalGeneration"]}))
    (base_dir / "tokenizer.json").write_text("{}")
    (base_dir / "tokenizer_config.json").write_text("{}")
    (base_dir / "processor_config.json").write_text("{}")
    (base_dir / "chat_template.jinja").write_text("")
    return tensors


def _write_fake_adapter(adapter_dir: Path, n_layers: int = 3, r: int = 2, alpha: int = 4) -> None:
    """Write a fake PEFT LoRA adapter targeting q_proj on each layer."""
    import torch
    from safetensors.torch import save_file

    adapter_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {}
    for i in range(n_layers):
        path = f"base_model.model.model.language_model.layers.{i}.self_attn.q_proj"
        tensors[f"{path}.lora_A.weight"] = torch.zeros(r, 4, dtype=torch.bfloat16)
        tensors[f"{path}.lora_B.weight"] = torch.zeros(4, r, dtype=torch.bfloat16)
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "modules_to_save": None})
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_quant_with_adapter_preserves_base_key_set(tmp_path, monkeypatch):
    """The merged dir produced by ``run_quant.main --adapter ...`` must
    contain every key from the base safetensors.

    This is the direct invariant the HF save_pretrained path violates
    on Gemma 4 E2B (k_proj/v_proj keys 15-34 silently dropped).
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "out"

    base_tensors = _write_fake_base(base_dir, n_layers=3)
    _write_fake_adapter(adapter_dir, n_layers=3)

    # Stub the dispatch so we never try to actually quantize.
    captured: dict = {}

    def fake_dispatch(method, merged_dir, out_dir, extra=None):
        captured["merged_dir"] = merged_dir
        captured["method"] = method
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir

    from scripts.run import quant as run_quant
    monkeypatch.setattr(run_quant, "dispatch", fake_dispatch)

    rc = run_quant.main(
        [
            "--method", "bnb_nf4",
            "--base_model", str(base_dir),
            "--adapter", str(adapter_dir),
            "--output_dir", str(output_dir),
        ]
    )
    assert rc == 0, "run_quant.main should exit 0"

    merged_dir = Path(captured["merged_dir"])
    merged_st = merged_dir / "model.safetensors"
    assert merged_st.is_file(), f"No model.safetensors at {merged_st}"

    with safe_open(str(merged_st), framework="pt") as f:
        merged_keys = set(f.keys())

    missing = set(base_tensors) - merged_keys
    assert not missing, (
        f"Merge dropped {len(missing)} base key(s): {sorted(missing)[:5]}. "
        "This is the KV-drop bug — run_quant must use merge_safetensors.merge."
    )


def test_run_quant_with_adapter_does_not_call_hf_load(tmp_path, monkeypatch):
    """Tripwire: ``load_bf16_multimodal`` (the HF transformers path)
    MUST NOT be called by run_quant when an adapter is provided.

    Documented bug: transformers v5.8 + PEFT silently drops
    k_proj/v_proj/k_norm/v_norm on Gemma 4 E2B layers 15-34.
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    base_dir = tmp_path / "base"
    adapter_dir = tmp_path / "adapter"
    output_dir = tmp_path / "out"

    _write_fake_base(base_dir)
    _write_fake_adapter(adapter_dir)

    def must_not_call(*args, **kwargs):
        raise AssertionError(
            "run_quant.main with --adapter called load_bf16_multimodal. "
            "It must use merge_safetensors.merge instead — the HF "
            "save_pretrained path silently drops K/V tensors on Gemma 4 E2B."
        )

    from src.common import model_io
    from scripts.run import quant as run_quant

    monkeypatch.setattr(model_io, "load_bf16_multimodal", must_not_call)
    monkeypatch.setattr(run_quant, "dispatch", lambda *a, **k: a[2])

    rc = run_quant.main(
        [
            "--method", "bnb_nf4",
            "--base_model", str(base_dir),
            "--adapter", str(adapter_dir),
            "--output_dir", str(output_dir),
        ]
    )
    assert rc == 0


def test_run_quant_without_adapter_passes_base_through(tmp_path, monkeypatch):
    """No-adapter path: ``run_quant.main --base_model ...`` should
    resolve the base dir and hand it to the dispatcher directly,
    NOT re-save via ``save_pretrained`` (which would also trigger the
    KV-drop on Gemma 4).
    """
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")

    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    _write_fake_base(base_dir)

    def must_not_call(*args, **kwargs):
        raise AssertionError(
            "run_quant.main without --adapter called load_bf16_multimodal. "
            "There's nothing to merge — pass the base dir through."
        )

    from src.common import model_io
    from scripts.run import quant as run_quant

    captured: dict = {}

    def fake_dispatch(method, merged_dir, out_dir, extra=None):
        captured["merged_dir"] = Path(merged_dir).resolve()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(model_io, "load_bf16_multimodal", must_not_call)
    monkeypatch.setattr(run_quant, "dispatch", fake_dispatch)

    rc = run_quant.main(
        [
            "--method", "bnb_nf4",
            "--base_model", str(base_dir),
            "--output_dir", str(output_dir),
        ]
    )
    assert rc == 0
    # base_dir is passed through to dispatch — same on-disk path.
    assert captured["merged_dir"] == base_dir.resolve()
