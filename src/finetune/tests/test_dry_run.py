"""End-to-end dry-run test: exercise finetune.py CLI without CUDA / unsloth."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


def _make_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    train = tmp_path / "data" / "train.jsonl"
    val = tmp_path / "data" / "val.jsonl"
    train.parent.mkdir(parents=True)

    train_records = [
        {
            "image": str(tmp_path / "img" / "a.jpg"),
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant", "content": "Eastern Hemlock."},
            ],
        },
        {
            "image": str(tmp_path / "img" / "b.jpg"),
            "conversations": [
                {"role": "user", "content": "Identify this."},
                {"role": "assistant", "content": "Sugar Maple."},
            ],
        },
    ]
    val_records = [train_records[0]]

    train.write_text("\n".join(json.dumps(r) for r in train_records) + "\n")
    val.write_text(json.dumps(val_records[0]) + "\n")

    config = tmp_path / "cfg.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
              max_seq_length: 256
            training:
              max_steps: 5
              num_train_epochs: null
              output_dir: "{tmp_path}/outputs"
            data:
              train_file: "{train}"
              val_file: "{val}"
            """
        )
    )
    return config, train, val


def test_dry_run_exits_zero_and_logs_stats(tmp_path: Path) -> None:
    config, _train, _val = _make_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "DRY RUN" in combined
    assert "Dry run complete" in combined
    assert "Skipped: FastModel.from_pretrained" in combined
    # Stats line for train should mention 2 records.
    assert "'records': 2" in combined


def test_dry_run_exits_zero_with_multi_val_files(tmp_path: Path) -> None:
    """Multi-val configs should log each validation partition in dry-run."""
    train = tmp_path / "data" / "train.jsonl"
    val_plant = tmp_path / "data" / "val_plant.jsonl"
    val_text = tmp_path / "data" / "val_text.jsonl"
    train.parent.mkdir(parents=True)

    image_record = {
        "image": str(tmp_path / "img" / "a.jpg"),
        "conversations": [
            {"role": "user", "content": "What plant is this?"},
            {"role": "assistant", "content": "Eastern Hemlock."},
        ],
    }
    text_record = {
        "conversations": [
            {"role": "user", "content": "Can you answer offline?"},
            {"role": "assistant", "content": "Yes."},
        ],
    }
    train.write_text(json.dumps(image_record) + "\n")
    val_plant.write_text(json.dumps(image_record) + "\n")
    val_text.write_text(json.dumps(text_record) + "\n")

    config = tmp_path / "cfg.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
              max_seq_length: 256
            training:
              max_steps: 5
              num_train_epochs: null
              output_dir: "{tmp_path}/outputs"
              modality_aware_sampler: true
            data:
              train_file: "{train}"
              val_file: null
              val_files:
                plant: "{val_plant}"
                text: "{val_text}"
            """
        )
    )

    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "val[plant]" in combined
    assert "val[text]" in combined


def test_dry_run_validation_errors_exit_nonzero(tmp_path: Path) -> None:
    """If config violates the freeze invariant, CLI should exit non-zero."""
    config = tmp_path / "bad.yaml"
    config.write_text(
        textwrap.dedent(
            """\
            lora:
              finetune_vision_layers: true
            data:
              train_file: "data/train.jsonl"
            """
        )
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "vision tower is frozen" in combined


# ---------------------------------------------------------------------------
# Projector tuning (feature/lora-plus-projector)
# ---------------------------------------------------------------------------


def _make_projector_fixture(tmp_path: Path) -> Path:
    """Same data shape as _make_fixture but enables tune_projector."""
    train = tmp_path / "data" / "train.jsonl"
    train.parent.mkdir(parents=True)
    train_records = [
        {
            "image": str(tmp_path / "img" / "a.jpg"),
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant", "content": "Eastern Hemlock."},
            ],
        },
    ]
    train.write_text("\n".join(json.dumps(r) for r in train_records) + "\n")

    config = tmp_path / "cfg.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
              max_seq_length: 256
              # Required: full-param projector tuning needs differentiable
              # base weights — 4-bit Linear4bit is not differentiable.
              load_in_4bit: false
            lora:
              tune_projector: true
              projector_learning_rate: 2.0e-5
            training:
              max_steps: 5
              num_train_epochs: null
              output_dir: "{tmp_path}/outputs"
            data:
              train_file: "{train}"
            """
        )
    )
    return config


def test_dry_run_projector_tuning_logs_plan(tmp_path: Path) -> None:
    """Dry-run with tune_projector=true should print a projector-aware
    freeze plan and exit 0 — without ever touching CUDA / unsloth."""
    config = _make_projector_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "DRY RUN" in combined

    # The dry-run must explicitly call out projector-tuning mode and
    # the candidate tokens it will look for at real-train time.
    # These two assertions only pass once finetune.py:dry_run is updated.
    assert "Projector tuning ENABLED" in combined, (
        "Expected projector-tuning banner in dry-run output but did not "
        "find it. Output:\n" + combined
    )
    assert "candidate token" in combined.lower(), (
        "Expected candidate token list in dry-run output. Output:\n" + combined
    )
    # Should mention the projector LR (auto-derived from training.lr / 10).
    assert "projector" in combined.lower() and "learning rate" in combined.lower()


def test_dry_run_lora_only_does_not_log_projector_banner(tmp_path: Path) -> None:
    """Backward-compat: when tune_projector is False (default), the
    dry-run output must NOT contain the projector-tuning banner."""
    config, _t, _v = _make_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    assert "Projector tuning ENABLED" not in combined


def test_dry_run_projector_with_finetune_vision_rejects(tmp_path: Path) -> None:
    """Validator rejects tune_projector=True + finetune_vision_layers=True."""
    train = tmp_path / "data" / "train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text(json.dumps({
        "image": str(tmp_path / "x.jpg"),
        "conversations": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
    }) + "\n")
    config = tmp_path / "bad.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            lora:
              tune_projector: true
              finetune_vision_layers: true
            data:
              train_file: "{train}"
            """
        )
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "tune_projector" in combined and "vision" in combined


# ---------------------------------------------------------------------------
# Vision-tower last-N tuning
# (feature/lora-plus-projector-plus-vision-tower)
# ---------------------------------------------------------------------------


def _make_vision_layer_fixture(tmp_path: Path) -> Path:
    """Enables tune_projector + tune_last_n_vision_layers."""
    train = tmp_path / "data" / "train.jsonl"
    train.parent.mkdir(parents=True)
    train_records = [
        {
            "image": str(tmp_path / "img" / "a.jpg"),
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant", "content": "Eastern Hemlock."},
            ],
        },
    ]
    train.write_text("\n".join(json.dumps(r) for r in train_records) + "\n")

    config = tmp_path / "cfg.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            model:
              base_model: "unsloth/gemma-4-E2B-it"
              max_seq_length: 256
              # Required: full-param projector + vision-layer tuning needs
              # differentiable base weights — Linear4bit is not.
              load_in_4bit: false
            lora:
              tune_projector: true
              projector_learning_rate: 5.0e-5
              tune_last_n_vision_layers: 2
              vision_layers_learning_rate: 1.0e-5
            training:
              max_steps: 5
              num_train_epochs: null
              output_dir: "{tmp_path}/outputs"
            data:
              train_file: "{train}"
            """
        )
    )
    return config


def test_dry_run_vision_layer_tuning_logs_plan(tmp_path: Path) -> None:
    """tune_last_n_vision_layers=2 should print a vision-tower-aware
    freeze plan, including the layer count and LR."""
    config = _make_vision_layer_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"dry-run failed: stdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    assert "DRY RUN" in combined
    # Plan must mention vision-tower tuning and the number of layers.
    lower = combined.lower()
    assert "vision" in lower and "tuning" in lower
    assert "last 2" in lower or "n=2" in lower or "2 vision encoder" in lower, (
        "Expected the freeze plan to mention the vision-layer count. "
        "Output:\n" + combined
    )
    # And the vision LR.
    assert "1.00e-05" in combined or "vision_layers_learning_rate" in combined or \
           "1e-05" in combined, (
        "Expected the vision_layers_learning_rate in dry-run output. "
        "Output:\n" + combined
    )


def test_dry_run_vision_layer_without_projector_rejects(tmp_path: Path) -> None:
    """Validator rejects tune_last_n_vision_layers > 0 without tune_projector."""
    train = tmp_path / "data" / "train.jsonl"
    train.parent.mkdir(parents=True)
    train.write_text(json.dumps({
        "image": str(tmp_path / "x.jpg"),
        "conversations": [{"role": "user", "content": "x"},
                          {"role": "assistant", "content": "y"}],
    }) + "\n")
    config = tmp_path / "bad.yaml"
    config.write_text(
        textwrap.dedent(
            f"""\
            lora:
              tune_projector: false
              tune_last_n_vision_layers: 2
            data:
              train_file: "{train}"
            """
        )
    )
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "tune_last_n_vision_layers" in combined
    assert "tune_projector" in combined


def test_dry_run_lora_only_does_not_log_vision_tower_banner(tmp_path: Path) -> None:
    """LoRA-only configs (no vision-layer tuning) must NOT mention the
    vision-tower tuning banner."""
    config, _t, _v = _make_fixture(tmp_path)
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "src.finetune", "--config", str(config), "--dry-run"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    combined = result.stdout + result.stderr
    # Old banner uses "Projector tuning ENABLED"; new banner uses similar
    # phrasing. We just check the vision-tower-specific text is absent.
    assert "vision encoder layer" not in combined.lower() or \
           "last 2" not in combined.lower()
