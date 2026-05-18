"""bitsandbytes NF4 — non-CUDA-dependent tests.

These tests verify the public surface of ``src.methods.bnb_nf4``
(config defaults, error paths, dispatch wiring). They do NOT load the
bf16 base model or invoke bitsandbytes; the actual quantization is
exercised separately as an integration smoke (requires CUDA + the
~9.5 GB bf16 model on disk).
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_config_defaults_match_qlora_era_recipe():
    """Defaults follow the QLoRA paper recipe: NF4 4-bit weights, bf16
    compute, double quantization on. Any drift here changes the eval
    matrix's reference 4-bit data point.
    """
    from src.methods.bnb_nf4 import BnBNF4Config

    cfg = BnBNF4Config()
    assert cfg.compute_dtype == "bfloat16"
    assert cfg.double_quant is True
    assert cfg.quant_storage_dtype == "uint8"


def test_quantize_rejects_missing_input_dir(tmp_path):
    """An obviously-wrong input path must fail loudly before any
    transformers / bitsandbytes import. This avoids 30 s of import
    overhead for a typo in a path argument.
    """
    from src.methods.bnb_nf4 import BnBNF4Config, quantize

    missing = tmp_path / "does_not_exist"
    output = tmp_path / "out"
    with pytest.raises((NotADirectoryError, FileNotFoundError)):
        quantize(missing, output, BnBNF4Config())


def test_dispatch_no_longer_stub(tmp_path):
    """``bnb_nf4`` is now implemented (PTQ output for the eval matrix).
    The dispatcher must route to the real ``quantize`` and not raise
    ``NotImplementedError`` for a valid (if non-existent) input dir;
    instead the failure mode should be the same as any other method
    given a bad path: ``NotADirectoryError`` / ``FileNotFoundError``.
    """
    from scripts.run.quant import dispatch

    missing = tmp_path / "merged_bf16_does_not_exist"
    output = tmp_path / "out"
    with pytest.raises((NotADirectoryError, FileNotFoundError)):
        dispatch("bnb_nf4", missing, output)


def test_build_bnb_config_maps_to_bitsandbytes_kwargs():
    """``_build_bnb_quantization_config`` should produce a
    ``BitsAndBytesConfig`` that matches the QLoRA-era recipe described
    in TODO Step 2b. Validated without invoking any quantization.
    """
    pytest.importorskip("bitsandbytes")
    from transformers import BitsAndBytesConfig

    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(BnBNF4Config())
    assert isinstance(qc, BitsAndBytesConfig)
    assert qc.load_in_4bit is True
    assert qc.bnb_4bit_quant_type == "nf4"
    assert qc.bnb_4bit_use_double_quant is True
    # compute_dtype is exposed under several names across transformers
    # versions; just confirm it's the bf16 torch dtype.
    import torch

    assert qc.bnb_4bit_compute_dtype == torch.bfloat16


def test_build_bnb_config_maps_storage_dtype():
    """``quant_storage_dtype`` is a public config knob and must not be
    silently ignored. bitsandbytes stores packed 4-bit values in an
    integer tensor by default; if a caller changes the storage dtype,
    the HF quantization config should reflect it.
    """
    pytest.importorskip("bitsandbytes")
    import torch

    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(BnBNF4Config(quant_storage_dtype="float16"))
    assert qc.bnb_4bit_quant_storage == torch.float16


def test_skip_modules_defaults_to_none():
    """Default behavior must remain "quantize every Linear" so the
    existing 0.1 % PlantNet baseline result is reproducible from
    `BnBNF4Config()` with no arguments.
    """
    from src.methods.bnb_nf4 import BnBNF4Config

    assert BnBNF4Config().skip_modules is None


def test_build_bnb_config_omits_skip_when_none():
    """``llm_int8_skip_modules`` is an OPTIONAL argument on
    ``BitsAndBytesConfig`` — passing ``None`` produces a config whose
    skip list is the bnb default (empty / no skips). We must not
    explicitly stamp ``None`` into the config when the user did not
    ask for it, because a future bnb version may treat explicit-None
    differently from "argument not supplied".
    """
    pytest.importorskip("bitsandbytes")
    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(BnBNF4Config())
    skip = getattr(qc, "llm_int8_skip_modules", None)
    # Either the attribute is missing entirely or it's the bnb default
    # (typically None or an empty list — both mean "skip nothing").
    assert skip is None or skip == []


def test_build_bnb_config_forwards_skip_modules_and_adds_lm_head():
    """When a caller passes ``skip_modules=[...]``, the resulting
    ``BitsAndBytesConfig.llm_int8_skip_modules`` must contain those
    substrings AND ``lm_head``. The ``lm_head`` auto-add is a
    correctness fix for tied-embedding models like Gemma 4 — see the
    docstring on ``_build_bnb_quantization_config``. Without it, the
    ``save_pretrained → from_pretrained`` round-trip silently
    produces a model that emits all-zero logits.
    """
    pytest.importorskip("bitsandbytes")
    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(
        BnBNF4Config(skip_modules=["embed_vision", "vision_tower"])
    )
    # Logical names are translated to real full-path prefixes that
    # transformers' ``should_convert_module`` actually matches.
    assert qc.llm_int8_skip_modules == [
        "model.embed_vision",
        "model.vision_tower",
        "lm_head",
    ]


def test_build_bnb_config_does_not_duplicate_lm_head():
    """If the caller already includes ``lm_head``, we must not produce
    duplicate entries (defensive — the bnb / transformers fast path
    walks this list per Linear, so a duplicate would be wasteful but
    not incorrect; the test is still cheap insurance against silent
    list-growth bugs)."""
    pytest.importorskip("bitsandbytes")
    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(
        BnBNF4Config(skip_modules=["embed_vision", "lm_head"])
    )
    # ``embed_vision`` translates to ``model.embed_vision``;
    # ``lm_head`` stays as-is (top-level direct child).
    assert qc.llm_int8_skip_modules == ["model.embed_vision", "lm_head"]


def test_skip_names_unknown_pass_through_unchanged():
    """Raw module-path prefixes that aren't in the logical-name map
    must pass through untouched, so future model architectures can
    use this knob without code changes."""
    pytest.importorskip("bitsandbytes")
    from src.methods.bnb_nf4 import (
        BnBNF4Config,
        _build_bnb_quantization_config,
    )

    qc = _build_bnb_quantization_config(
        BnBNF4Config(
            skip_modules=["model.encoder.layers.31", "some.exotic.path"]
        )
    )
    assert qc.llm_int8_skip_modules == [
        "model.encoder.layers.31",
        "some.exotic.path",
        "lm_head",
    ]


def test_resolve_skip_names_function():
    """Direct unit-test of the resolver — it's a small pure function
    and easy to test in isolation."""
    from src.methods.bnb_nf4 import _resolve_skip_names

    assert _resolve_skip_names(["vision_tower"]) == ["model.vision_tower"]
    assert _resolve_skip_names(["lm_head"]) == ["lm_head"]
    assert _resolve_skip_names(["unknown.path"]) == ["unknown.path"]
    # Order is preserved.
    assert _resolve_skip_names(["embed_vision", "vision_tower"]) == [
        "model.embed_vision",
        "model.vision_tower",
    ]


def test_run_quant_threads_skip_modules_through_dispatch(monkeypatch, tmp_path):
    """``run_quant.dispatch`` must forward ``--bnb_skip_modules`` into
    the ``BnBNF4Config(skip_modules=...)`` it builds, otherwise the
    CLI flag would silently no-op and we'd waste a 45-minute ablation
    run quantizing the wrong scope.
    """
    captured: dict = {}

    def fake_quantize(input_dir, output_dir, config):
        captured["config"] = config
        return output_dir

    import src.methods.bnb_nf4 as bnb_mod

    monkeypatch.setattr(bnb_mod, "quantize", fake_quantize)

    from scripts.run.quant import dispatch

    merged = tmp_path / "merged"
    merged.mkdir()
    out = tmp_path / "out"
    dispatch(
        "bnb_nf4",
        merged,
        out,
        extra={"bnb_skip_modules": ["embed_vision"]},
    )
    assert captured["config"].skip_modules == ["embed_vision"]
