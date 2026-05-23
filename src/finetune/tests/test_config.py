"""Tests for src.config — defaults, YAML overlay, CLI overrides, validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.config import (
    FinetuneConfig,
    apply_cli_overrides,
    load_config,
    validate_config,
)


def test_defaults_match_spec() -> None:
    cfg = FinetuneConfig()
    # Model default tracks the iOS app's checkpoint family.
    assert cfg.model.base_model == "unsloth/gemma-4-E2B-it"
    assert cfg.model.max_seq_length == 1024
    # Project policy: NO QLoRA. Default is full bf16 precision throughout.
    assert cfg.model.load_in_4bit is False
    # bf16 is the recommended compute dtype for Gemma 4: native on
    # Ampere+ and no fp16 attention softmax overflow. Locked in
    # explicitly so older-GPU auto-detect doesn't silently downgrade.
    assert cfg.model.dtype == "bfloat16"
    # Vision + audio MUST be off by default.
    assert cfg.lora.finetune_vision_layers is False
    assert cfg.lora.finetune_audio_layers is False
    assert cfg.lora.finetune_language_layers is True


def test_load_config_yaml_overlay(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
              max_seq_length: 2048
            lora:
              r: 16
              lora_alpha: 32
            training:
              learning_rate: 1.0e-5
              max_steps: 100
            """
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.model.max_seq_length == 2048
    assert cfg.lora.r == 16
    assert cfg.lora.lora_alpha == 32
    assert cfg.training.learning_rate == pytest.approx(1.0e-5)
    assert cfg.training.max_steps == 100


def test_yaml_overlay_dtype_bfloat16(tmp_path: Path) -> None:
    """YAML `dtype: bfloat16` resolves to the string 'bfloat16'
    (passed verbatim to FastModel.from_pretrained at runtime)."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("model:\n  dtype: bfloat16\n")
    cfg = load_config(yaml_path)
    assert cfg.model.dtype == "bfloat16"


def test_yaml_overlay_dtype_null_overrides_to_none(tmp_path: Path) -> None:
    """Explicit `dtype: null` in YAML lets users opt out of bf16 default
    and fall back to unsloth's auto-detect (bf16 on Ampere+, fp16 on
    V100/T4). Verify None is round-tripped, not silently kept as bf16."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("model:\n  dtype: null\n")
    cfg = load_config(yaml_path)
    assert cfg.model.dtype is None


def test_yaml_overlay_dtype_float16(tmp_path: Path) -> None:
    """Users can still pick fp16 explicitly (e.g. for V100). No
    validator override — bf16 is the default but not enforced."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("model:\n  dtype: float16\n")
    cfg = load_config(yaml_path)
    assert cfg.model.dtype == "float16"


def test_unknown_yaml_keys_are_warnings_not_errors(tmp_path: Path, caplog) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("model:\n  not_a_real_field: 42\n")
    cfg = load_config(yaml_path)  # must not raise
    assert cfg.model.base_model == "unsloth/gemma-4-E2B-it"
    assert any("Unknown config key" in rec.message for rec in caplog.records)


def test_cli_overrides_take_precedence(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("training:\n  learning_rate: 1.0e-5\n")
    cfg = load_config(
        yaml_path,
        cli_overrides={"learning_rate": 7.5e-4, "lora_r": 64},
    )
    assert cfg.training.learning_rate == pytest.approx(7.5e-4)
    assert cfg.lora.r == 64


def test_cli_override_none_values_ignored() -> None:
    cfg = FinetuneConfig()
    apply_cli_overrides(cfg, {"learning_rate": None, "lora_r": 99})
    assert cfg.training.learning_rate == FinetuneConfig().training.learning_rate
    assert cfg.lora.r == 99


def test_validate_rejects_unfrozen_vision() -> None:
    cfg = FinetuneConfig()
    cfg.lora.finetune_vision_layers = True
    errors = validate_config(cfg)
    assert any("vision tower is frozen" in e for e in errors)


def test_validate_rejects_unfrozen_audio() -> None:
    cfg = FinetuneConfig()
    cfg.lora.finetune_audio_layers = True
    errors = validate_config(cfg)
    assert any("audio tower is frozen" in e for e in errors)


def test_validate_requires_epochs_or_steps() -> None:
    cfg = FinetuneConfig()
    cfg.training.num_train_epochs = None
    cfg.training.max_steps = None
    errors = validate_config(cfg)
    assert any("num_train_epochs" in e or "max_steps" in e for e in errors)


def test_validate_passes_for_default_config() -> None:
    cfg = FinetuneConfig()
    assert validate_config(cfg) == []


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_no_yaml() -> None:
    """No YAML path == pure defaults."""
    cfg = load_config(None)
    assert cfg == FinetuneConfig()


# ---------------------------------------------------------------------------
# Projector tuning (feature/lora-plus-projector)
# ---------------------------------------------------------------------------


def test_default_tune_projector_is_false() -> None:
    """Default behavior must be unchanged from feature/finetune-unsloth:
    projector tuning is opt-in, off by default."""
    cfg = FinetuneConfig()
    assert cfg.lora.tune_projector is False
    assert cfg.lora.projector_learning_rate is None


def test_validator_rejects_tune_projector_with_finetune_vision_layers() -> None:
    """Tuning the projector while ALSO unfreezing the encoder is a
    contradiction this branch explicitly does not support."""
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.finetune_vision_layers = True
    errors = validate_config(cfg)
    assert any("tune_projector" in e and "vision" in e for e in errors)


def test_validator_rejects_negative_projector_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.projector_learning_rate = -1e-5
    errors = validate_config(cfg)
    assert any("projector_learning_rate" in e for e in errors)


def test_validator_rejects_zero_projector_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.projector_learning_rate = 0.0
    errors = validate_config(cfg)
    assert any("projector_learning_rate" in e for e in errors)


def test_validator_accepts_positive_projector_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = False  # redundant w/ default; explicit for clarity
    cfg.lora.tune_projector = True
    cfg.lora.projector_learning_rate = 2e-5
    assert validate_config(cfg) == []


def test_validator_accepts_tune_projector_with_unset_lr() -> None:
    """projector_learning_rate=None means 'auto = training.lr / 10' at runtime."""
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = False  # redundant w/ default; explicit for clarity
    cfg.lora.tune_projector = True
    cfg.lora.projector_learning_rate = None
    assert validate_config(cfg) == []


def test_cli_override_tune_projector_and_lr() -> None:
    cfg = FinetuneConfig()
    apply_cli_overrides(
        cfg,
        {"tune_projector": True, "projector_learning_rate": 5e-5},
    )
    assert cfg.lora.tune_projector is True
    assert cfg.lora.projector_learning_rate == pytest.approx(5e-5)


def test_yaml_overlay_tune_projector(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            lora:
              tune_projector: true
              projector_learning_rate: 2.0e-5
            """
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.lora.tune_projector is True
    assert cfg.lora.projector_learning_rate == pytest.approx(2.0e-5)


# ---------------------------------------------------------------------------
# QLoRA policy: default is full bf16 precision. Explicit QLoRA configs may
# opt in with model.load_in_4bit=true, but the validator must warn so
# accidental 4-bit training is visible.
# ---------------------------------------------------------------------------


def test_default_load_in_4bit_is_false() -> None:
    """Project policy: default config starts in bf16 LoRA mode."""
    cfg = FinetuneConfig()
    assert cfg.model.load_in_4bit is False


def test_validator_warns_load_in_4bit_pure_lora(caplog) -> None:
    """Explicit QLoRA opt-in is allowed but must be visible in logs."""
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = True
    # No projector / vision tuning — just plain LoRA on a 4-bit base.
    assert cfg.lora.tune_projector is False
    assert cfg.lora.tune_last_n_vision_layers == 0
    with caplog.at_level("WARNING"):
        errors = validate_config(cfg)
    assert errors == []
    assert "model.load_in_4bit=True" in caplog.text
    assert "QLoRA" in caplog.text


def test_validator_allows_load_in_4bit_with_tune_projector_and_warns(caplog) -> None:
    """Projector QLoRA is guarded later by the trainable-4bit tripwire."""
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = True
    cfg.lora.tune_projector = True
    with caplog.at_level("WARNING"):
        errors = validate_config(cfg)
    assert errors == []
    assert "model.load_in_4bit=True" in caplog.text


def test_validator_allows_load_in_4bit_with_tune_last_n_vision_layers_and_warns(caplog) -> None:
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = True
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    with caplog.at_level("WARNING"):
        errors = validate_config(cfg)
    assert errors == []
    assert "model.load_in_4bit=True" in caplog.text


def test_validator_rejects_dora_with_qlora() -> None:
    """DoRA + QLoRA is an invalid combination."""
    cfg = FinetuneConfig()
    cfg.lora.use_dora = True
    cfg.model.load_in_4bit = True
    errors = validate_config(cfg)
    assert any("use_dora" in e and "load_in_4bit" in e for e in errors)


def test_validator_accepts_dora_bf16() -> None:
    """DoRA in bf16 mode (no QLoRA) is valid."""
    cfg = FinetuneConfig()
    cfg.lora.use_dora = True
    assert cfg.model.load_in_4bit is False
    assert validate_config(cfg) == []


def test_validator_accepts_default_full_precision() -> None:
    """Default config (load_in_4bit=False, no projector) passes."""
    cfg = FinetuneConfig()
    assert cfg.model.load_in_4bit is False
    assert validate_config(cfg) == []


@pytest.mark.parametrize(
    "optim_name",
    [
        "adamw_8bit",
        "adamw_torch_8bit",
        "adamw_torch_4bit",
        "ademamix_8bit",
        "paged_ademamix_8bit",
        "galore_adamw_8bit",
        "rmsprop_bnb_8bit",
    ],
)
def test_validator_rejects_any_quantized_optimizer_name(optim_name: str) -> None:
    """Project policy bans quantized optimizer states, not just bnb AdamW8bit."""
    cfg = FinetuneConfig()
    cfg.training.optim = optim_name
    errors = validate_config(cfg)
    assert any(optim_name in e and "8-bit / 4-bit" in e for e in errors), errors


def test_validator_accepts_fp_with_tune_projector_and_vision_layers() -> None:
    """Full-precision base + projector + vision-tower tuning is the
    intended path for feature/lora-plus-projector-plus-vision-tower."""
    cfg = FinetuneConfig()
    # load_in_4bit=False is now the default, but be explicit for clarity.
    cfg.model.load_in_4bit = False
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    assert validate_config(cfg) == []


# ---------------------------------------------------------------------------
# Vision-tower last-N tuning (feature/lora-plus-projector-plus-vision-tower)
# ---------------------------------------------------------------------------


def test_default_tune_last_n_vision_layers_is_zero() -> None:
    """Default behavior must be unchanged: vision-tower tuning is opt-in, off."""
    cfg = FinetuneConfig()
    assert cfg.lora.tune_last_n_vision_layers == 0
    assert cfg.lora.vision_layers_learning_rate is None


def test_validator_rejects_negative_tune_last_n_vision_layers() -> None:
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True  # would be required anyway
    cfg.lora.tune_last_n_vision_layers = -1
    errors = validate_config(cfg)
    assert any("tune_last_n_vision_layers" in e for e in errors)


def test_validator_rejects_tune_vision_layers_without_tune_projector() -> None:
    """Unfreezing vision encoder layers without also tuning the projector
    creates a feature-space misalignment the language LoRA can't fix."""
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = False
    cfg.lora.tune_last_n_vision_layers = 2
    errors = validate_config(cfg)
    assert any(
        "tune_last_n_vision_layers" in e and "tune_projector" in e for e in errors
    )


def test_validator_rejects_tune_vision_layers_with_finetune_vision_layers() -> None:
    """Last-N vision layer tuning is incompatible with all-vision-layer
    finetuning — they're competing mechanisms."""
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    cfg.lora.finetune_vision_layers = True
    errors = validate_config(cfg)
    # Either the existing finetune_vision_layers error fires, or a new
    # tune_last_n_vision_layers-specific one. We accept either.
    assert any("finetune_vision_layers" in e or "vision tower" in e for e in errors)


def test_validator_rejects_negative_vision_layers_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    cfg.lora.vision_layers_learning_rate = -1e-5
    errors = validate_config(cfg)
    assert any("vision_layers_learning_rate" in e for e in errors)


def test_validator_rejects_zero_vision_layers_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    cfg.lora.vision_layers_learning_rate = 0.0
    errors = validate_config(cfg)
    assert any("vision_layers_learning_rate" in e for e in errors)


def test_validator_accepts_positive_vision_layers_learning_rate() -> None:
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = False  # redundant w/ default; explicit for clarity
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    cfg.lora.vision_layers_learning_rate = 1e-5
    assert validate_config(cfg) == []


def test_validator_accepts_tune_vision_layers_with_unset_lr() -> None:
    """vision_layers_learning_rate=None means 'auto = training.lr / 20' at runtime."""
    cfg = FinetuneConfig()
    cfg.model.load_in_4bit = False  # redundant w/ default; explicit for clarity
    cfg.lora.tune_projector = True
    cfg.lora.tune_last_n_vision_layers = 2
    cfg.lora.vision_layers_learning_rate = None
    assert validate_config(cfg) == []


def test_validator_accepts_zero_n_with_projector_off() -> None:
    """tune_last_n_vision_layers=0 means off — must not require tune_projector."""
    cfg = FinetuneConfig()
    cfg.lora.tune_projector = False
    cfg.lora.tune_last_n_vision_layers = 0
    assert validate_config(cfg) == []


def test_cli_override_tune_vision_layers_and_lr() -> None:
    cfg = FinetuneConfig()
    apply_cli_overrides(
        cfg,
        {"tune_last_n_vision_layers": 2, "vision_layers_learning_rate": 1e-5},
    )
    assert cfg.lora.tune_last_n_vision_layers == 2
    assert cfg.lora.vision_layers_learning_rate == pytest.approx(1e-5)


def test_yaml_overlay_tune_vision_layers(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            lora:
              tune_projector: true
              tune_last_n_vision_layers: 2
              vision_layers_learning_rate: 1.0e-5
            """
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.lora.tune_projector is True
    assert cfg.lora.tune_last_n_vision_layers == 2
    assert cfg.lora.vision_layers_learning_rate == pytest.approx(1.0e-5)


# ---------------------------------------------------------------------------
# EvalConfig — auto-eval after training (default behaviour as of 2026-05-11)
# ---------------------------------------------------------------------------


def test_eval_config_defaults() -> None:
    """Auto-eval is ON by default; sane defaults match the v1 manual eval workflow."""
    cfg = FinetuneConfig()
    assert cfg.eval.enabled is True
    assert cfg.eval.max_eval_samples == 300
    assert cfg.eval.max_new_tokens == 256
    assert cfg.eval.use_unsloth is True
    assert cfg.eval.batch_size == 1


def test_eval_config_yaml_overlay(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            eval:
              enabled: false
              max_eval_samples: 50
              max_new_tokens: 128
              use_unsloth: false
              batch_size: 4
            """
        )
    )
    cfg = load_config(yaml_path)
    assert cfg.eval.enabled is False
    assert cfg.eval.max_eval_samples == 50
    assert cfg.eval.max_new_tokens == 128
    assert cfg.eval.use_unsloth is False
    assert cfg.eval.batch_size == 4


def test_eval_config_yaml_overlay_partial(tmp_path: Path) -> None:
    """Partial overlays preserve unspecified defaults."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("eval:\n  max_eval_samples: 500\n")
    cfg = load_config(yaml_path)
    assert cfg.eval.max_eval_samples == 500
    # Unspecified fields keep their defaults.
    assert cfg.eval.enabled is True
    assert cfg.eval.max_new_tokens == 256
    assert cfg.eval.use_unsloth is True


def test_eval_config_max_eval_samples_can_be_null(tmp_path: Path) -> None:
    """max_eval_samples=null means 'evaluate all val samples'."""
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("eval:\n  max_eval_samples: null\n")
    cfg = load_config(yaml_path)
    assert cfg.eval.max_eval_samples is None


def test_cli_override_eval_enabled_and_samples() -> None:
    cfg = FinetuneConfig()
    apply_cli_overrides(
        cfg,
        {"eval_enabled": False, "max_eval_samples_eval": 25},
    )
    assert cfg.eval.enabled is False
    assert cfg.eval.max_eval_samples == 25


def test_validator_rejects_zero_max_new_tokens() -> None:
    cfg = FinetuneConfig()
    cfg.eval.max_new_tokens = 0
    errors = validate_config(cfg)
    assert any("max_new_tokens" in e for e in errors)


def test_validator_rejects_negative_batch_size() -> None:
    cfg = FinetuneConfig()
    cfg.eval.batch_size = -1
    errors = validate_config(cfg)
    assert any("batch_size" in e for e in errors)


def test_validator_accepts_max_eval_samples_none() -> None:
    cfg = FinetuneConfig()
    cfg.eval.max_eval_samples = None
    assert validate_config(cfg) == []


def test_validator_rejects_zero_max_eval_samples() -> None:
    """0 is meaningless — either set a positive limit or use null for 'all'."""
    cfg = FinetuneConfig()
    cfg.eval.max_eval_samples = 0
    errors = validate_config(cfg)
    assert any("max_eval_samples" in e for e in errors)
