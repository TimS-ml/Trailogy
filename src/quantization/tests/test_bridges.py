"""Tests for ``src.bridges`` placeholders.

The bridge implementations are not done yet (B.1.3 is the next
deliverable). These tests pin two things:

1. The module surface is importable (the placeholder file exists and
   is discoverable by name).
2. The placeholders raise ``NotImplementedError`` with a message
   pointing to the design doc / roadmap section — so callers don't
   silently get a no-op when they try to use the bridge today.
"""

from __future__ import annotations

import pytest


def test_hf_gptq_to_mlx_module_imports():
    from src.bridges import hf_gptq_to_mlx

    assert hasattr(hf_gptq_to_mlx, "bridge")
    assert hasattr(hf_gptq_to_mlx, "naive_bridge")


def test_bridge_raises_not_implemented(tmp_path):
    from src.bridges.hf_gptq_to_mlx import bridge

    with pytest.raises(NotImplementedError, match="B.1.3"):
        bridge(tmp_path / "in", tmp_path / "out")


def test_naive_bridge_raises_not_implemented(tmp_path):
    from src.bridges.hf_gptq_to_mlx import naive_bridge

    with pytest.raises(NotImplementedError, match="B.1.3"):
        naive_bridge(tmp_path / "in", tmp_path / "out")
