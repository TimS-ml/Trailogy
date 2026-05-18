"""Tests for the mac_mlx_lm preflight guards.

The mlx_lm GPTQ / AWQ / DWQ / dynamic_quant wrappers all hit known
upstream bugs on Gemma 4. The roadmap §"Why B.1 over B.2" enumerates
them; these tests pin the matching code-side guards:

- ``KNOWN_UPSTREAM_BUGS`` covers all four wrappers.
- ``warn_known_bug`` logs the message at WARNING level.
- ``awq.assert_supports_model`` raises ``NotImplementedError`` (before
  any mlx_lm load) when the model_type is missing from
  ``AWQ_MODEL_CONFIGS``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


def test_known_upstream_bugs_covers_all_wrappers():
    from src.methods.mac_mlx_lm import KNOWN_UPSTREAM_BUGS

    for name in ("gptq", "awq", "dwq", "dynamic_quant"):
        assert name in KNOWN_UPSTREAM_BUGS
        # Each entry should be a non-trivial human message, not empty.
        assert len(KNOWN_UPSTREAM_BUGS[name]) > 20


def test_warn_known_bug_logs_at_warning(caplog):
    from src.methods.mac_mlx_lm import warn_known_bug

    with caplog.at_level(logging.WARNING, logger="src.methods.mac_mlx_lm"):
        warn_known_bug("gptq")
    assert any("known upstream bug" in r.message for r in caplog.records)


def test_warn_known_bug_no_log_for_unknown_wrapper(caplog):
    """Don't log noise for wrapper names not on the bug list."""
    from src.methods.mac_mlx_lm import warn_known_bug

    with caplog.at_level(logging.WARNING, logger="src.methods.mac_mlx_lm"):
        warn_known_bug("not_a_real_wrapper")
    assert not any("known upstream bug" in r.message for r in caplog.records)


def test_read_model_type_returns_value(tmp_path):
    from src.methods.mac_mlx_lm import read_model_type

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    assert read_model_type(tmp_path) == "gemma4"


def test_read_model_type_returns_none_on_missing(tmp_path):
    from src.methods.mac_mlx_lm import read_model_type

    assert read_model_type(tmp_path) is None


def test_awq_assert_supports_model_rejects_gemma4(tmp_path, monkeypatch):
    """When AWQ_MODEL_CONFIGS exists but lacks gemma4, the guard must
    raise — and the error message must point at the roadmap context.
    """
    pytest.importorskip("mlx_lm.quant.awq", reason="mlx_lm.quant.awq not installed")

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gemma4"}))

    from src.methods.mac_mlx_lm import awq as awq_mod

    # Force a known mapping so the test outcome is deterministic
    # regardless of installed mlx_lm version.
    fake_configs = {"llama": object(), "mistral": object()}
    monkeypatch.setattr(
        "mlx_lm.quant.awq.AWQ_MODEL_CONFIGS", fake_configs, raising=False
    )

    with pytest.raises(NotImplementedError, match="AWQ_MODEL_CONFIGS"):
        awq_mod.assert_supports_model(tmp_path)


def test_awq_assert_supports_model_passes_when_supported(tmp_path, monkeypatch):
    pytest.importorskip("mlx_lm.quant.awq", reason="mlx_lm.quant.awq not installed")

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    from src.methods.mac_mlx_lm import awq as awq_mod

    fake_configs = {"llama": object()}
    monkeypatch.setattr(
        "mlx_lm.quant.awq.AWQ_MODEL_CONFIGS", fake_configs, raising=False
    )

    awq_mod.assert_supports_model(tmp_path)  # no raise


def test_awq_assert_supports_model_silent_when_no_config(tmp_path):
    """No config.json → can't determine model_type → don't raise.
    Upstream will fail loudly with whatever its own error is."""
    from src.methods.mac_mlx_lm import awq as awq_mod

    awq_mod.assert_supports_model(tmp_path)  # no raise
