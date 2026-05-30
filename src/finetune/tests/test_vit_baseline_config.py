"""Tests for vit_baseline.config — defaults, YAML overlay, overrides, validation."""

from __future__ import annotations

import textwrap

import pytest

from vit_baseline.config import (
    VitBaselineConfig,
    apply_cli_overrides,
    load_config,
    resolve_image_root,
    validate_config,
)


def test_defaults() -> None:
    cfg = VitBaselineConfig()
    assert cfg.model.backbone == "vit_base_patch16_siglip_224"
    assert cfg.model.freeze_backbone is False
    # Project policy: bf16 only.
    assert cfg.train.dtype == "bfloat16"
    assert cfg.train.learning_rate > 0


def test_yaml_overlay(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(textwrap.dedent("""
        model:
          backbone: convnext_base
          freeze_backbone: true
        train:
          num_train_epochs: 3
          learning_rate: 1.0e-3
    """))
    cfg = load_config(p)
    assert cfg.model.backbone == "convnext_base"
    assert cfg.model.freeze_backbone is True
    assert cfg.train.num_train_epochs == 3
    assert cfg.train.learning_rate == pytest.approx(1.0e-3)


def test_cli_overrides_take_precedence() -> None:
    cfg = VitBaselineConfig()
    apply_cli_overrides(cfg, {"backbone": "eva02_base_patch14_448", "learning_rate": 5e-4})
    assert cfg.model.backbone == "eva02_base_patch14_448"
    assert cfg.train.learning_rate == pytest.approx(5e-4)
    # None values are ignored (no clobber).
    apply_cli_overrides(cfg, {"backbone": None})
    assert cfg.model.backbone == "eva02_base_patch14_448"


def test_validate_rejects_8bit_optim() -> None:
    cfg = VitBaselineConfig()
    cfg.train.optim = "adamw_8bit"
    errs = validate_config(cfg)
    assert any("8-bit" in e for e in errs)


def test_validate_rejects_bad_dtype() -> None:
    cfg = VitBaselineConfig()
    cfg.train.dtype = "int8"
    assert any("dtype" in e for e in validate_config(cfg))


def test_valid_config_has_no_errors() -> None:
    assert validate_config(VitBaselineConfig()) == []


def test_resolve_image_root_from_env(monkeypatch) -> None:
    cfg = VitBaselineConfig()
    monkeypatch.setenv("PLANT_IMAGE_ROOT", "/some/root")
    assert resolve_image_root(cfg) == "/some/root"
    # Config value wins over env.
    cfg.data.image_root = "/explicit"
    assert resolve_image_root(cfg) == "/explicit"


def test_resolve_image_root_missing_raises(monkeypatch) -> None:
    monkeypatch.delenv("PLANT_IMAGE_ROOT", raising=False)
    with pytest.raises(ValueError):
        resolve_image_root(VitBaselineConfig())
