"""Tests for src.evaluate's --config mode.

`--config <yaml>` lets `evaluate.py` pull its defaults from the same
FinetuneConfig the training run used. Explicit CLI flags still win.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest

from src.config import FinetuneConfig, load_config
from src.evaluate import (
    apply_config_defaults,
    apply_fallback_defaults,
    build_parser,
)


# ---------------------------------------------------------------------------
# apply_config_defaults — populate Namespace from FinetuneConfig
# ---------------------------------------------------------------------------


def _ns(**kwargs):
    """Helper: build a Namespace with all config-overridable fields = None."""
    base = dict(
        base_model=None,
        adapter_path=None,
        test_file=None,
        output_file=None,
        run_name=None,
        max_eval_samples=None,
        max_new_tokens=None,
        use_unsloth=None,
        batch_size=None,
        quantize=None,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_apply_config_defaults_fills_base_model_from_model_section() -> None:
    cfg = FinetuneConfig()
    cfg.model.base_model = "unsloth/gemma-4-E2B-it"
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.base_model == "unsloth/gemma-4-E2B-it"


def test_apply_config_defaults_derives_adapter_path_from_output_dir() -> None:
    cfg = FinetuneConfig()
    cfg.training.output_dir = "outputs/my-run_20260511_000000"
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.adapter_path == "outputs/my-run_20260511_000000/final-adapter"


def test_apply_config_defaults_derives_adapter_path_from_run_name_when_available() -> None:
    """Manual auto-eval reruns pass --run_name but often omit --adapter_path.

    Training rewrites output_dir to outputs/{run_name}; raw YAML output_dir can
    be stale, so run_name must win when deriving the adapter path.
    """
    cfg = FinetuneConfig()
    cfg.training.output_dir = "outputs/static-config-dir"
    args = _ns(run_name="actual-run_20260512_120000")
    apply_config_defaults(args, cfg)
    assert args.adapter_path == "outputs/actual-run_20260512_120000/final-adapter"


def test_apply_config_defaults_fills_test_file_from_data_val_file() -> None:
    cfg = FinetuneConfig()
    cfg.data.val_file = "data/val.jsonl"
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.test_file == "data/val.jsonl"


def test_apply_config_defaults_fills_run_name_from_training() -> None:
    cfg = FinetuneConfig()
    cfg.training.run_name = "my-run_20260511_000000"
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.run_name == "my-run_20260511_000000"


def test_apply_config_defaults_fills_eval_section_fields() -> None:
    cfg = FinetuneConfig()
    cfg.eval.max_eval_samples = 200
    cfg.eval.max_new_tokens = 256
    cfg.eval.use_unsloth = True
    cfg.eval.batch_size = 1
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.max_eval_samples == 200
    assert args.max_new_tokens == 256
    assert args.use_unsloth is True
    assert args.batch_size == 1


def test_explicit_cli_args_override_config_values() -> None:
    """If an args field is already non-None, config must NOT overwrite it."""
    cfg = FinetuneConfig()
    cfg.model.base_model = "unsloth/gemma-4-E2B-it"
    cfg.eval.use_unsloth = True
    cfg.eval.max_new_tokens = 256
    args = _ns(
        base_model="custom/other-model",
        use_unsloth=False,
        max_new_tokens=128,
    )
    apply_config_defaults(args, cfg)
    assert args.base_model == "custom/other-model"
    assert args.use_unsloth is False
    assert args.max_new_tokens == 128


def test_apply_config_defaults_preserves_explicit_max_eval_samples_zero_is_explicit() -> None:
    """Subtle: 0 must be treated as explicit (user disabled), not None.

    Implementation must check `is None`, not falsy. (Config validator
    rejects 0 anyway, but the helper itself must not coerce.)
    """
    cfg = FinetuneConfig()
    cfg.eval.max_eval_samples = 200
    args = _ns(max_eval_samples=0)
    apply_config_defaults(args, cfg)
    assert args.max_eval_samples == 0


def test_apply_config_defaults_max_eval_samples_null_in_config_is_propagated() -> None:
    """null in yaml -> None in cfg -> None in args (means 'all samples')."""
    cfg = FinetuneConfig()
    cfg.eval.max_eval_samples = None
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.max_eval_samples is None


# ---------------------------------------------------------------------------
# apply_fallback_defaults — fill remaining None for no-config invocations
# ---------------------------------------------------------------------------


def test_apply_fallback_defaults_when_no_config() -> None:
    """Backwards-compat: invocation without --config still gets sane defaults."""
    args = _ns()
    apply_fallback_defaults(args)
    # These mirror the pre-config-mode behaviour of the script.
    assert args.base_model == "google/gemma-4-e2b-it"
    assert args.max_eval_samples == 300
    assert args.max_new_tokens == 256
    assert args.use_unsloth is False
    assert args.batch_size == 1


def test_apply_fallback_defaults_keeps_existing_values() -> None:
    args = _ns(base_model="x", max_new_tokens=99, use_unsloth=True, batch_size=4)
    apply_fallback_defaults(args)
    assert args.base_model == "x"
    assert args.max_new_tokens == 99
    assert args.use_unsloth is True
    assert args.batch_size == 4


def test_eval_config_max_eval_samples_default_300() -> None:
    """Project default: routine eval uses a 300-record cap."""
    cfg = FinetuneConfig()
    assert cfg.eval.max_eval_samples == 300


def test_apply_fallback_defaults_can_preserve_config_null_max_eval_samples() -> None:
    """YAML `eval.max_eval_samples: null` remains the explicit full-set escape hatch."""
    args = _ns(max_eval_samples=None)
    apply_fallback_defaults(args, preserve_max_eval_samples_none=True)
    assert args.max_eval_samples is None


# ---------------------------------------------------------------------------
# build_parser — argparse surface after the refactor
# ---------------------------------------------------------------------------


def test_parser_has_config_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/foo.yaml", "--test_file", "x"])
    assert args.config == "configs/foo.yaml"


def test_parser_config_overridable_args_default_to_none() -> None:
    """All config-overridable args must default to None so apply_config_defaults
    can detect 'user did not specify' robustly."""
    parser = build_parser()
    args = parser.parse_args(["--test_file", "x"])
    assert args.base_model is None
    assert args.max_new_tokens is None
    assert args.use_unsloth is None
    assert args.batch_size is None
    # Already None pre-refactor — guard against regression.
    assert args.adapter_path is None
    assert args.run_name is None
    assert args.max_eval_samples is None


def test_parser_use_unsloth_boolean_optional() -> None:
    """`--use_unsloth` and `--no-use_unsloth` must both flip the flag."""
    parser = build_parser()
    on = parser.parse_args(["--use_unsloth", "--test_file", "x"])
    off = parser.parse_args(["--no-use_unsloth", "--test_file", "x"])
    assert on.use_unsloth is True
    assert off.use_unsloth is False


def test_parser_quantize_defaults_to_none_for_config_fallback() -> None:
    """Without --quantize / --no-quantize, args.quantize must be None so
    apply_config_defaults can fall back to cfg.eval.load_in_4bit."""
    parser = build_parser()
    args = parser.parse_args(["--test_file", "x"])
    assert args.quantize is None


def test_parser_quantize_boolean_optional() -> None:
    """`--quantize` and `--no-quantize` must both flip the flag."""
    parser = build_parser()
    on = parser.parse_args(["--quantize", "--test_file", "x"])
    off = parser.parse_args(["--no-quantize", "--test_file", "x"])
    assert on.quantize is True
    assert off.quantize is False


# ---------------------------------------------------------------------------
# Eval load_in_4bit default + config wiring
# ---------------------------------------------------------------------------


def test_eval_config_load_in_4bit_default_false() -> None:
    """Project policy: eval defaults to bf16, matching training dtype.
    Inference 4-bit is opt-in for memory-constrained eval boxes."""
    cfg = FinetuneConfig()
    assert cfg.eval.load_in_4bit is False


def test_apply_config_defaults_fills_quantize_from_eval_section() -> None:
    """When --quantize is not on the CLI (args.quantize=None),
    cfg.eval.load_in_4bit drives the value."""
    cfg = FinetuneConfig()
    cfg.eval.load_in_4bit = True
    args = _ns()
    apply_config_defaults(args, cfg)
    assert args.quantize is True


def test_apply_config_defaults_respects_explicit_cli_quantize() -> None:
    """Explicit --quantize / --no-quantize wins over config."""
    cfg = FinetuneConfig()
    cfg.eval.load_in_4bit = True
    args = _ns(quantize=False)  # user passed --no-quantize
    apply_config_defaults(args, cfg)
    assert args.quantize is False


def test_apply_fallback_defaults_quantize_off_when_no_config() -> None:
    """Bare CLI invocation (no --config) → default off."""
    args = _ns()
    apply_fallback_defaults(args)
    assert args.quantize is False


def test_eval_yaml_overlay_load_in_4bit(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("eval:\n  load_in_4bit: true\n")
    cfg = load_config(yaml_path)
    assert cfg.eval.load_in_4bit is True


# ---------------------------------------------------------------------------
# End-to-end: yaml load + apply_config_defaults
# ---------------------------------------------------------------------------


def test_e2e_yaml_drives_args(tmp_path: Path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        textwrap.dedent(
            """\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
            training:
              output_dir: "outputs/e2e_run"
              run_name: "e2e_run"
            data:
              val_file: "data/val.jsonl"
            eval:
              max_eval_samples: 50
              max_new_tokens: 128
              use_unsloth: true
              batch_size: 2
              load_in_4bit: false
            """
        )
    )
    cfg = load_config(yaml_path)
    args = _ns()
    apply_config_defaults(args, cfg)
    apply_fallback_defaults(args)  # no-op for fields filled by config
    assert args.base_model == "unsloth/gemma-4-E2B-it"
    assert args.adapter_path == "outputs/e2e_run/final-adapter"
    assert args.test_file == "data/val.jsonl"
    assert args.run_name == "e2e_run"
    assert args.max_eval_samples == 50
    assert args.max_new_tokens == 128
    assert args.use_unsloth is True
    assert args.batch_size == 2
    assert args.quantize is False  # explicit eval.load_in_4bit: false
