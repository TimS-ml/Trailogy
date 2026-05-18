"""Tests for the v3 ``regularization`` config block.

Covers:
  - Defaults: disabled (bit-identical to existing training).
  - YAML overlay parses correctly.
  - Validator rejects negative weights / temperatures.
  - Mutual exclusion / interaction with QLoRA path (KL teacher forward
    under disable_adapter() requires the base weights to be queryable,
    which is fine under bf16 LoRA; explicitly note QLoRA caveat).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import FinetuneConfig, RegularizationConfig, load_config, validate_config


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def test_regularization_config_defaults_are_disabled() -> None:
    """Defaults are bit-identical-to-v2: no regularization unless opted in
    explicitly via YAML."""
    rc = RegularizationConfig()
    assert rc.kl_enabled is False
    assert rc.l2_enabled is False
    # Sensible-but-conservative numbers, in case the user only flips
    # enabled=true and forgets to set the weight.
    assert rc.kl_weight > 0.0
    assert rc.kl_temperature == 1.0
    assert rc.l2_weight > 0.0


def test_finetune_config_has_regularization_field() -> None:
    cfg = FinetuneConfig()
    assert hasattr(cfg, "regularization")
    assert isinstance(cfg.regularization, RegularizationConfig)


# ---------------------------------------------------------------------------
# YAML overlay
# ---------------------------------------------------------------------------


def test_yaml_overlay_regularization_block(tmp_path: Path) -> None:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "regularization:\n"
        "  kl_enabled: true\n"
        "  kl_weight: 0.05\n"
        "  kl_temperature: 2.0\n"
        "  l2_enabled: true\n"
        "  l2_weight: 1.0e-4\n"
    )
    cfg = load_config(p)
    assert cfg.regularization.kl_enabled is True
    assert cfg.regularization.kl_weight == pytest.approx(0.05)
    assert cfg.regularization.kl_temperature == pytest.approx(2.0)
    assert cfg.regularization.l2_enabled is True
    assert cfg.regularization.l2_weight == pytest.approx(1.0e-4)


def test_yaml_partial_overlay_preserves_other_defaults(tmp_path: Path) -> None:
    """A YAML that sets only kl_enabled must leave the L2 fields at default."""
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "regularization:\n"
        "  kl_enabled: true\n"
    )
    cfg = load_config(p)
    assert cfg.regularization.kl_enabled is True
    assert cfg.regularization.l2_enabled is False  # default preserved


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def test_validator_accepts_disabled_defaults() -> None:
    cfg = FinetuneConfig()
    errors = validate_config(cfg)
    # Should not introduce new errors; default v1 config remains valid
    # apart from the existing 'num_train_epochs or max_steps' rule.
    reg_errors = [e for e in errors if "regularization" in e]
    assert reg_errors == []


@pytest.mark.parametrize("bad", [0.0, -0.1, -1.0])
def test_validator_rejects_non_positive_kl_weight(bad: float) -> None:
    cfg = FinetuneConfig()
    cfg.regularization.kl_enabled = True
    cfg.regularization.kl_weight = bad
    cfg.training.max_steps = 10  # satisfy unrelated rule
    errors = validate_config(cfg)
    assert any("regularization.kl_weight" in e for e in errors)


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_validator_rejects_non_positive_kl_temperature(bad: float) -> None:
    cfg = FinetuneConfig()
    cfg.regularization.kl_enabled = True
    cfg.regularization.kl_temperature = bad
    cfg.training.max_steps = 10
    errors = validate_config(cfg)
    assert any("regularization.kl_temperature" in e for e in errors)


@pytest.mark.parametrize("bad", [0.0, -1e-6])
def test_validator_rejects_non_positive_l2_weight(bad: float) -> None:
    cfg = FinetuneConfig()
    cfg.regularization.l2_enabled = True
    cfg.regularization.l2_weight = bad
    cfg.training.max_steps = 10
    errors = validate_config(cfg)
    assert any("regularization.l2_weight" in e for e in errors)


def test_validator_disabled_weight_value_ignored() -> None:
    """When kl_enabled=False, a zero/negative kl_weight must NOT raise
    (the value is dead code in that case). This keeps reduce-to-zero
    'turn it off' configs viable without rewriting the numeric.
    """
    cfg = FinetuneConfig()
    cfg.regularization.kl_enabled = False
    cfg.regularization.kl_weight = 0.0
    cfg.training.max_steps = 10
    errors = validate_config(cfg)
    assert not any("regularization.kl_weight" in e for e in errors)


def test_validator_rejects_regularization_without_modality_aware_sampler() -> None:
    """The compute_loss hook lives on ModalityAwareSFTTrainer. Plain
    SFTTrainer would silently drop the regularizers — fail loud at
    config time."""
    cfg = FinetuneConfig()
    cfg.regularization.kl_enabled = True
    cfg.training.modality_aware_sampler = False
    cfg.training.max_steps = 10
    errors = validate_config(cfg)
    assert any("modality_aware_sampler" in e for e in errors)


def test_validator_accepts_regularization_with_modality_aware_sampler() -> None:
    cfg = FinetuneConfig()
    cfg.regularization.kl_enabled = True
    cfg.regularization.l2_enabled = True
    cfg.training.modality_aware_sampler = True
    cfg.training.max_steps = 10
    errors = validate_config(cfg)
    assert not any("modality_aware_sampler" in e for e in errors)


def test_validator_warns_kl_with_qlora(caplog) -> None:
    """KL teacher forward = ``model(input)`` under disable_adapter(). With
    QLoRA the base weights are 4-bit, so the teacher distribution
    reflects a quantization-distorted base rather than the true
    pretrained base. The validator should warn (not hard-reject) so
    users can opt in if they really mean it.
    """
    import logging

    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = True
    cfg.regularization.kl_enabled = True
    cfg.training.max_steps = 10
    with caplog.at_level(logging.WARNING):
        validate_config(cfg)
    msgs = " ".join(rec.message for rec in caplog.records)
    assert "kl" in msgs.lower() and ("qlora" in msgs.lower() or "4bit" in msgs.lower() or "4-bit" in msgs.lower())
