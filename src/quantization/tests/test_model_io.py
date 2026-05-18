"""``src.common.model_io`` — non-CUDA-dependent tests.

These tests verify ``save_bf16_merged`` writes processor / tokenizer
side-cars alongside the model weights. Without these files in the
merged dir, downstream tools (``gptqmodel.GPTQModel.load``,
``mlx_vlm.convert``, ``AutoProcessor.from_pretrained``) fail loudly.

We use fake stand-ins (no real transformers model load) so the test
runs in <100 ms with no GPU and no HF network access.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest


class _FakeModel:
    """Minimal stand-in for ``Gemma4ForConditionalGeneration``."""

    def __init__(self, name_or_path: str = "") -> None:
        self.config = MagicMock()
        self.config.name_or_path = name_or_path

    def save_pretrained(self, output_dir, **kwargs):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}")
        (out / "model.safetensors").write_bytes(b"\x00" * 8)


class _FakeProcessor:
    """Stand-in for an ``AutoProcessor`` instance."""

    def __init__(self, source: str = "") -> None:
        self.source = source
        self.save_pretrained_calls: list[Path] = []

    def save_pretrained(self, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "tokenizer.json").write_text("{}")
        (out / "tokenizer_config.json").write_text("{}")
        (out / "processor_config.json").write_text("{}")
        (out / "chat_template.jinja").write_text("")
        self.save_pretrained_calls.append(out)


def test_save_bf16_merged_writes_processor_when_source_given(tmp_path):
    """Regression for the 2026-05-13 GPTQ crash: gptqmodel.load() calls
    AutoTokenizer.from_pretrained(merged_dir) and fails when tokenizer
    side-cars are missing."""
    from src.common import model_io

    fake_model = _FakeModel()
    fake_processor = _FakeProcessor(source="fake/source")
    out = tmp_path / "merged"

    with mock.patch.object(model_io, "_load_processor", return_value=fake_processor) as load_p:
        model_io.save_bf16_merged(fake_model, out, processor_source="fake/source")

    load_p.assert_called_once_with("fake/source")
    assert fake_processor.save_pretrained_calls == [out]
    assert (out / "tokenizer.json").is_file()
    assert (out / "tokenizer_config.json").is_file()
    assert (out / "processor_config.json").is_file()
    assert (out / "model.safetensors").is_file()


def test_save_bf16_merged_fallback_copies_from_base(tmp_path):
    """When no processor_source is given, save_bf16_merged should fall
    back to copying side-cars from model.config.name_or_path."""
    from src.common.model_io import save_bf16_merged

    base = tmp_path / "base"
    base.mkdir()
    (base / "tokenizer.json").write_text('{"tok": true}')
    (base / "tokenizer_config.json").write_text('{"cfg": true}')
    (base / "processor_config.json").write_text('{"proc": true}')
    (base / "chat_template.jinja").write_text("tmpl")
    (base / "special_tokens_map.json").write_text('{"sp": true}')

    fake_model = _FakeModel(name_or_path=str(base))

    out = tmp_path / "merged"
    save_bf16_merged(fake_model, out)

    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()
    assert (out / "processor_config.json").exists()
    assert (out / "chat_template.jinja").exists()


def test_save_bf16_merged_fallback_does_not_overwrite(tmp_path):
    """Fallback copy must not overwrite files already written by
    save_pretrained (e.g. config.json)."""
    from src.common.model_io import save_bf16_merged

    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"base": true}')
    (base / "tokenizer.json").write_text('{"tok": true}')

    fake_model = _FakeModel(name_or_path=str(base))

    out = tmp_path / "merged"
    save_bf16_merged(fake_model, out)

    # config.json must be the one from save_pretrained, not from base
    assert '{}' in (out / "config.json").read_text()
    assert (out / "tokenizer.json").exists()


def test_save_bf16_merged_warns_when_no_processor_source(tmp_path, caplog):
    """When no processor_source and no resolvable base dir, warn."""
    from src.common import model_io

    fake_model = _FakeModel(name_or_path="/nonexistent/path")
    out = tmp_path / "merged_no_proc"

    with caplog.at_level("WARNING", logger="src.common.model_io"):
        model_io.save_bf16_merged(fake_model, out)

    assert (out / "model.safetensors").is_file()
    warning_text = " ".join(r.message for r in caplog.records)
    assert "processor" in warning_text.lower() or "tokenizer" in warning_text.lower()
    assert not (out / "tokenizer.json").exists()


def test_save_bf16_merged_returns_output_dir(tmp_path):
    """API contract: returns the (created) output dir as a Path."""
    from src.common import model_io

    fake_model = _FakeModel()
    out = tmp_path / "merged_return"

    result = model_io.save_bf16_merged(fake_model, out)
    assert isinstance(result, Path)
    assert result == out
    assert result.is_dir()


def test_load_processor_uses_autoprocessor(tmp_path):
    """``_load_processor`` goes through ``AutoProcessor.from_pretrained``."""
    from src.common import model_io

    fake_proc = _FakeProcessor(source="anywhere")
    with mock.patch(
        "transformers.AutoProcessor.from_pretrained", return_value=fake_proc
    ) as af:
        got = model_io._load_processor("some/repo-id")

    af.assert_called_once_with("some/repo-id", trust_remote_code=True)
    assert got is fake_proc


# ---------------------------------------------------------------------------
# copy_processor_assets — shared helper for PTQ methods that emit
# their own model files (gptqmodel.save_quantized, bnb's save_pretrained)
# but DON'T preserve multimodal processor side-cars.
# ---------------------------------------------------------------------------

# All processor/tokenizer side-cars we mirror. Sourced from the bnb_nf4
# implementation that has been exercising this list since 2026-05-12.
_EXPECTED_FILES = {
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "generation_config.json",
}


def test_copy_processor_assets_mirrors_only_present_files(tmp_path):
    """The helper must copy every side-car that exists in ``src``, and
    silently skip ones that don't (e.g. ``tokenizer.model`` is absent
    on Gemma 4 fast-tokenizer checkpoints — should not raise).
    """
    from src.common import model_io

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()

    # Only a subset of the side-cars exist (mirrors the real
    # _merged_bf16 layout: processor_config but no preprocessor_config,
    # no tokenizer.model, no special_tokens_map).
    present = {"tokenizer.json", "tokenizer_config.json",
               "processor_config.json", "chat_template.jinja",
               "generation_config.json"}
    for name in present:
        (src / name).write_text("PAYLOAD-" + name)

    copied = model_io.copy_processor_assets(src, dst)

    assert set(copied) == present
    for name in present:
        assert (dst / name).read_text() == "PAYLOAD-" + name
    # No spurious files (didn't copy ones that don't exist):
    assert set(p.name for p in dst.iterdir()) == present


def test_copy_processor_assets_returns_empty_when_src_has_nothing(tmp_path):
    """Returns ``[]`` if the source dir has no relevant side-cars;
    no exception. This is what makes the helper safe to call
    unconditionally from PTQ methods.
    """
    from src.common import model_io

    src = tmp_path / "src_empty"
    dst = tmp_path / "dst_empty"
    src.mkdir()
    dst.mkdir()

    copied = model_io.copy_processor_assets(src, dst)
    assert copied == []
    assert list(dst.iterdir()) == []


def test_copy_processor_assets_covers_multimodal_processor_config(tmp_path):
    """Regression for the 2026-05-13 GPTQ eval crash: ``gptqmodel.save_quantized``
    writes ``tokenizer.json`` but not ``processor_config.json``, so
    ``AutoProcessor.from_pretrained(out_dir)`` fails on multimodal
    checkpoints. The helper MUST list ``processor_config.json`` as one
    of the files it copies.
    """
    from src.common import model_io

    src = tmp_path / "src_mm"
    dst = tmp_path / "dst_mm"
    src.mkdir()
    dst.mkdir()
    (src / "processor_config.json").write_text("{}")

    copied = model_io.copy_processor_assets(src, dst)
    assert "processor_config.json" in copied
    assert (dst / "processor_config.json").is_file()
