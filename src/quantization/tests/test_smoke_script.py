"""Smoke script static checks.

The script is intentionally bash because it glues together several CLI
smokes, but it still needs one important invariant: all Python entrypoints
must use the same interpreter. Otherwise a shell with the wrong PATH can
run tests against a package set that lacks torch/bitsandbytes while later
commands run somewhere else.
"""

from __future__ import annotations

from pathlib import Path


def test_smoke_script_uses_one_python_entrypoint():
    # Path relative to this test file (quantization/tests/) so the test
    # is independent of pytest cwd.
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run/smoke_4090.sh"
    script = script_path.read_text()

    assert "PYTHON_BIN=" in script
    assert "\"$PYTHON_BIN\" -m pytest" in script
    assert "\"$PYTHON_BIN\" -m src.methods.gptq" in script
    assert "\"$PYTHON_BIN\" -m scripts.inspect.quantized" in script
    assert "\"$PYTHON_BIN\" -m scripts.run.quant" in script

    forbidden = [
        "    pytest ",
        "    python -m ",
        "      python -m ",
    ]
    for needle in forbidden:
        assert needle not in script, f"bare Python entrypoint still present: {needle!r}"


def test_smoke_script_uses_post_move_test_path():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run/smoke_4090.sh"
    script = script_path.read_text()

    assert '"$PYTHON_BIN" -m pytest tests/ -q' in script
    assert "pytest quantization/tests/" not in script


def test_experiment_scripts_source_shared_env_helper():
    scripts_dir = Path(__file__).resolve().parent.parent / "scripts" / "run"

    for name in ("experiments_local_4090.sh", "experiments_laptop_4090.sh"):
        script = (scripts_dir / name).read_text()
        assert 'source "$SCRIPT_DIR/../_env/_common.sh"' in script
        assert 'source "$SCRIPT_DIR/_common.sh"' not in script


def test_common_env_defaults_follow_src_layout():
    common = (
        Path(__file__).resolve().parent.parent / "scripts" / "_env" / "_common.sh"
    ).read_text()

    assert "$REPO_ROOT/../finetune/data/val.jsonl" in common
    assert "$REPO_ROOT/../finetune/data/train.jsonl" in common
    assert "$REPO_ROOT/results" in common
    assert "$REPO_ROOT/finetune/data" not in common
    assert "$REPO_ROOT/quantization/results" not in common


def test_mlx_env_has_no_local_conda_path_default():
    mlx_env = (
        Path(__file__).resolve().parent.parent / "scripts" / "_env" / "_mlx_env.sh"
    ).read_text()

    assert "miniforge3" not in mlx_env
    assert "AGENTS.md" not in mlx_env
