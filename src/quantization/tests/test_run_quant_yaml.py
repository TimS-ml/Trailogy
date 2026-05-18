"""Tests for ``run_quant.py``'s YAML config loading path.

Pins three contracts:

1. ``--config <yaml>`` is loaded and per-method sections are forwarded
   to the dispatcher under ``extra["config_yaml"]``.
2. YAML values flow into the dataclass init for the target method
   (e.g. ``gptq.group_size`` ends up in the constructed ``GPTQConfig``).
3. ``method:`` may come from the YAML when ``--method`` is omitted on
   the CLI; mismatch between the two raises a clear error.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


def _write_yaml(path: Path, body: str) -> None:
    path.write_text(dedent(body).lstrip("\n"))


def _make_fake_base(tmp_path: Path) -> Path:
    """Minimal local dir with the files merge_safetensors._resolve_base_dir
    needs to treat as a base. No real model weights.
    """
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    (base / "tokenizer.json").write_text("{}")
    return base


def test_yaml_method_used_when_cli_method_omitted(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    _write_yaml(cfg, """
        method: gptq
        gptq:
          bits: 4
          group_size: 128
    """)
    base = _make_fake_base(tmp_path)

    captured: dict = {}
    from scripts.run import quant as run_quant

    def fake_dispatch(method, merged_dir, out_dir, extra=None):
        captured["method"] = method
        captured["extra"] = extra or {}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir

    monkeypatch.setattr(run_quant, "dispatch", fake_dispatch)

    rc = run_quant.main(
        [
            "--config", str(cfg),
            "--base_model", str(base),
            "--output_dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 0
    assert captured["method"] == "gptq"
    assert captured["extra"]["config_yaml"]["gptq"]["group_size"] == 128


def test_yaml_cli_method_mismatch_errors(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    _write_yaml(cfg, """
        method: gptq
    """)
    base = _make_fake_base(tmp_path)

    from scripts.run import quant as run_quant
    rc = run_quant.main(
        [
            "--method", "bnb_nf4",
            "--config", str(cfg),
            "--base_model", str(base),
            "--output_dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_dispatch_passes_yaml_gptq_overrides_to_config(tmp_path, monkeypatch):
    """Direct dispatch() call: yaml_cfg under ``extra["config_yaml"]``
    must end up in ``GPTQConfig(**filtered)``.
    """
    captured: dict = {}

    # Patch GPTQConfig + quantize at the module level to capture kwargs.
    from src.methods import gptq as gptq_mod
    from scripts.run.quant import dispatch

    real_cfg_cls = gptq_mod.GPTQConfig

    def fake_quantize(merged_dir, output_dir, config, plantnet_jsonl=None):
        captured["config"] = config
        captured["plantnet_jsonl"] = plantnet_jsonl
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return output_dir

    monkeypatch.setattr(gptq_mod, "quantize", fake_quantize)

    out = tmp_path / "out"
    dispatch(
        "gptq",
        merged_dir=tmp_path / "merged",
        output_dir=out,
        extra={
            "config_yaml": {
                "method": "gptq",
                "gptq": {
                    "bits": 4,
                    "group_size": 64,  # NON-default — must propagate
                    "desc_act": True,
                    "future_unknown_field": 42,  # silently dropped
                },
            },
            "plantnet_jsonl": tmp_path / "train.jsonl",
        },
    )

    cfg = captured["config"]
    assert isinstance(cfg, real_cfg_cls)
    assert cfg.group_size == 64
    assert cfg.desc_act is True
    assert captured["plantnet_jsonl"] == tmp_path / "train.jsonl"


def test_dispatch_mlx_vlm_yaml_group_size_conflict_errors(tmp_path):
    """The mlx_vlm_g{N} method name pins group_size; YAML conflicts
    must fail rather than silently running a different recipe.
    """
    from scripts.run.quant import dispatch

    with pytest.raises(ValueError, match="q_group_size.*mlx_vlm_g64"):
        dispatch(
            "mlx_vlm_g64",
            merged_dir=tmp_path / "merged",
            output_dir=tmp_path / "out",
            extra={
                "config_yaml": {
                    "method": "mlx_vlm_g64",
                    "mlx_vlm": {"q_group_size": 999},
                },
            },
        )


def test_main_mlx_vlm_yaml_group_size_conflict_returns_2(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    _write_yaml(cfg, """
        method: mlx_vlm_g64
        mlx_vlm:
          q_group_size: 32
    """)
    base = _make_fake_base(tmp_path)

    from scripts.run import quant as run_quant

    rc = run_quant.main(
        [
            "--config", str(cfg),
            "--base_model", str(base),
            "--output_dir", str(tmp_path / "out"),
        ]
    )
    assert rc == 2


def test_dispatch_qat_export_accepts_yaml_recipe_path(tmp_path):
    """qat_export is a stub, but YAML should still reach its config
    object so dispatch fails with the intended NotImplementedError.
    """
    from scripts.run.quant import dispatch

    with pytest.raises(NotImplementedError, match="QAT export pending"):
        dispatch(
            "qat_export",
            merged_dir=tmp_path / "merged",
            output_dir=tmp_path / "out",
            extra={
                "config_yaml": {
                    "method": "qat_export",
                    "qat_export": {"qat_recipe_path": str(tmp_path / "recipe.json")},
                }
            },
        )
