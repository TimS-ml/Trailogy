"""Tests for v2 config additions: modality_aware_sampler, val_files dict,
eval_strategy / eval_steps / per_device_eval_batch_size.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import FinetuneConfig, load_config, validate_config


# ---------------------------------------------------------------------------
# Default values (backward compat)
# ---------------------------------------------------------------------------

def test_modality_aware_sampler_default_is_false():
    """Backward compat: existing configs that don't set this flag must
    behave exactly as before."""
    cfg = FinetuneConfig()
    assert cfg.training.modality_aware_sampler is False


def test_eval_strategy_default_is_no():
    """v1 default: no mid-training eval. v2 configs opt-in to 'steps'."""
    cfg = FinetuneConfig()
    assert cfg.training.eval_strategy == "no"
    assert cfg.training.eval_steps is None
    # Per-device eval batch size mirrors train default unless set.
    assert cfg.training.per_device_eval_batch_size is None


def test_data_val_files_default_is_none():
    """v1 configs use a single val_file (string). v2 configs use val_files
    (dict). Both should be supported."""
    cfg = FinetuneConfig()
    assert cfg.data.val_files is None
    assert cfg.data.val_file is not None  # v1 default unchanged


# ---------------------------------------------------------------------------
# YAML overlay
# ---------------------------------------------------------------------------

def test_yaml_overlay_modality_aware_sampler(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "training:\n"
        "  modality_aware_sampler: true\n"
        "  eval_strategy: steps\n"
        "  eval_steps: 500\n"
        "  per_device_eval_batch_size: 4\n"
    )
    cfg = load_config(p)
    assert cfg.training.modality_aware_sampler is True
    assert cfg.training.eval_strategy == "steps"
    assert cfg.training.eval_steps == 500
    assert cfg.training.per_device_eval_batch_size == 4


def test_yaml_overlay_val_files_dict(tmp_path: Path):
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "data:\n"
        "  train_file: data/mix-100k/train.jsonl\n"
        "  val_files:\n"
        "    plant:    data/mix-100k/val_plant.jsonl\n"
        "    nonplant: data/mix-100k/val_nonplant.jsonl\n"
        "    negative: data/mix-100k/val_negative.jsonl\n"
    )
    cfg = load_config(p)
    assert cfg.data.val_files == {
        "plant":    "data/mix-100k/val_plant.jsonl",
        "nonplant": "data/mix-100k/val_nonplant.jsonl",
        "negative": "data/mix-100k/val_negative.jsonl",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_rejects_modality_aware_sampler_with_group_by_length():
    """The two features collide: modality_aware_sampler does its OWN
    within-modality length sort, so HF's group_by_length must be off."""
    cfg = FinetuneConfig()
    cfg.training.modality_aware_sampler = True
    cfg.training.group_by_length = True
    errors = validate_config(cfg)
    assert any("group_by_length" in e.lower() for e in errors), (
        f"expected group_by_length conflict error, got: {errors}"
    )


def test_validate_accepts_modality_aware_sampler_with_group_by_length_off():
    cfg = FinetuneConfig()
    cfg.training.modality_aware_sampler = True
    cfg.training.group_by_length = False
    errors = validate_config(cfg)
    # Filter to only errors related to our flags.
    relevant = [e for e in errors if "modality" in e.lower() or "group_by_length" in e.lower()]
    assert relevant == [], f"unexpected errors: {relevant}"


def test_validate_rejects_eval_steps_with_strategy_no():
    cfg = FinetuneConfig()
    cfg.training.eval_strategy = "no"
    cfg.training.eval_steps = 500
    errors = validate_config(cfg)
    assert any("eval_steps" in e.lower() and "strategy" in e.lower() for e in errors)


def test_validate_requires_eval_steps_when_strategy_is_steps():
    cfg = FinetuneConfig()
    cfg.training.eval_strategy = "steps"
    cfg.training.eval_steps = None
    errors = validate_config(cfg)
    assert any("eval_steps" in e.lower() for e in errors)


def test_validate_accepts_eval_strategy_epoch():
    cfg = FinetuneConfig()
    cfg.training.eval_strategy = "epoch"
    errors = validate_config(cfg)
    relevant = [e for e in errors if "eval_strategy" in e.lower() or "eval_steps" in e.lower()]
    assert relevant == [], f"unexpected errors: {relevant}"


def test_validate_rejects_invalid_eval_strategy():
    cfg = FinetuneConfig()
    cfg.training.eval_strategy = "every_other_tuesday"
    errors = validate_config(cfg)
    assert any("eval_strategy" in e.lower() for e in errors)


def test_validate_rejects_val_files_dict_with_val_file_string():
    """A config can have val_files (v2 dict) OR val_file (v1 string),
    but not both — that signals the user is unsure which is active."""
    cfg = FinetuneConfig()
    cfg.data.val_file = "data/val.jsonl"
    cfg.data.val_files = {"plant": "data/val_plant.jsonl"}
    errors = validate_config(cfg)
    assert any("val_files" in e.lower() and "val_file" in e.lower() for e in errors)
