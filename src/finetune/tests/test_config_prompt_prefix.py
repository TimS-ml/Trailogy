"""Tests for the ``data.prompt_prefixes`` config field (camera-state gate)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import FinetuneConfig, load_config


def test_prompt_prefixes_default_is_none() -> None:
    """Backward compat: existing v2 configs that don't set this field
    must behave exactly as before (no prefix injected anywhere)."""
    cfg = FinetuneConfig()
    assert cfg.data.prompt_prefixes is None


def test_yaml_overlay_prompt_prefixes(tmp_path: Path) -> None:
    """A YAML overlay declaring both camera_on and camera_off prefixes
    round-trips into the DataConfig dict unchanged."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data:\n"
        "  prompt_prefixes:\n"
        "    camera_on:  \"[camera=on] \"\n"
        "    camera_off: \"[camera=off] \"\n"
    )
    cfg = load_config(p)
    assert cfg.data.prompt_prefixes is not None
    assert cfg.data.prompt_prefixes["camera_on"] == "[camera=on] "
    assert cfg.data.prompt_prefixes["camera_off"] == "[camera=off] "


def test_yaml_overlay_prompt_prefixes_camera_on_only(tmp_path: Path) -> None:
    """Asymmetric configs are valid — e.g. an ablation that only gates
    the vision branch. The missing key falls through to "no prefix" at
    dispatch time."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data:\n"
        "  prompt_prefixes:\n"
        "    camera_on: \"[camera=on] \"\n"
    )
    cfg = load_config(p)
    assert cfg.data.prompt_prefixes == {"camera_on": "[camera=on] "}
    assert "camera_off" not in cfg.data.prompt_prefixes
