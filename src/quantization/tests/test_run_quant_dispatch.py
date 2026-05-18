"""Test that the run_quant dispatcher knows every method in METHOD_REGISTRY
and routes to the right module — without actually invoking quantization
(which needs CUDA / MLX / large model weights).
"""

from __future__ import annotations

import pytest

from scripts.run.quant import METHOD_REGISTRY


def test_registry_includes_known_methods():
    expected = {
        "mlx_vlm_g32",
        "mlx_vlm_g64",
        "mlx_vlm_g128",
        "unsloth_ud",
        "gptq",
        "awq",
        "bnb_nf4",
        "qat_export",
    }
    # Allow registry to grow; just ensure these are all present.
    missing = expected - METHOD_REGISTRY
    assert not missing, f"Missing methods in registry: {missing}"


def test_unknown_method_raises_value_error(tmp_path):
    from scripts.run.quant import dispatch

    with pytest.raises(ValueError, match="Unknown method"):
        dispatch("does_not_exist", tmp_path, tmp_path)


def test_stub_methods_raise_not_implemented(tmp_path):
    """Methods that are intentionally stubs should raise
    NotImplementedError — not silently return None or empty.

    Note: GPTQ is implemented but requires `gptqmodel` + a real bf16
    model to actually run. ``bnb_nf4`` is implemented (PTQ output for
    the eval matrix; see ``test_bnb_nf4.py``). We exclude both here.
    The remaining stubs are waiting on either external work
    (``unsloth_ud`` diff) or are intentional low-priority placeholders
    (``awq``).
    """
    from scripts.run.quant import dispatch

    for stub_method in ["unsloth_ud", "awq"]:
        with pytest.raises(NotImplementedError):
            dispatch(stub_method, tmp_path, tmp_path)


def test_qat_export_requires_recipe_path(tmp_path):
    from scripts.run.quant import dispatch

    # No qat_recipe_path → ValueError.
    with pytest.raises(ValueError, match="qat_recipe_path"):
        dispatch("qat_export", tmp_path, tmp_path, extra={})
