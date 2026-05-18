"""Tests for ``src.metrics_callback.JsonlMetricsCallback``.

We don't pull in real HF Trainer here — only the public TrainerCallback
contract: ``on_log(args, state, control, logs=None, **kw)`` and
``on_train_end(...)``. The callback is exercised against tiny dummy
state objects and a tmp_path output dir; assertions check the JSONL
file is well-formed, line-per-emit, and carries the metadata fields
(``step``, ``epoch``, ``kind``) downstream consumers depend on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.metrics_callback import JsonlMetricsCallback, _classify_log


# ---------------------------------------------------------------------------
# Stand-ins for HF Trainer's args / state / control objects
# ---------------------------------------------------------------------------


@dataclass
class _FakeState:
    """Mirrors the attrs JsonlMetricsCallback reads from HF TrainerState."""

    global_step: int = 0
    epoch: float = 0.0


class _FakeArgs:  # noqa: D401 — placeholder, callback doesn't touch it
    pass


class _FakeControl:  # noqa: D401 — callback returns whatever we pass in
    pass


# ---------------------------------------------------------------------------
# _classify_log: stable kind tagging across the three HF log flavours
# ---------------------------------------------------------------------------


def test_classify_train_log_has_loss_and_lr() -> None:
    """Standard train-step log → ``"train"``."""
    logs = {"loss": 1.5, "learning_rate": 2e-4, "grad_norm": 0.3, "epoch": 0.1}
    assert _classify_log(logs) == "train"


def test_classify_eval_log_has_eval_prefix() -> None:
    """Any ``eval_*`` key wins → ``"eval"`` (matches HF multi-eval format)."""
    logs = {
        "eval_plant_loss": 0.8,
        "eval_plant_runtime": 12.5,
        "eval_nonplant_loss": 1.2,
    }
    assert _classify_log(logs) == "eval"


def test_classify_eval_takes_priority_when_both_present() -> None:
    """Edge case: HF sometimes emits eval + train keys in the same dict
    (eval at logging boundary). Eval wins so downstream filtering sees
    the eval cadence, not phantom train rows."""
    logs = {"loss": 1.0, "learning_rate": 1e-4, "eval_plant_loss": 0.9}
    assert _classify_log(logs) == "eval"


def test_classify_other_for_end_of_train_summary() -> None:
    """``train_runtime`` summary at end-of-training → ``"other"``."""
    logs = {"train_runtime": 3600.0, "train_samples_per_second": 4.5}
    assert _classify_log(logs) == "other"


# ---------------------------------------------------------------------------
# JsonlMetricsCallback: file I/O contract
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_callback_creates_file_on_first_log(tmp_path: Path) -> None:
    """File is created lazily, not on construction. A callback that's
    instantiated but never sees a log shouldn't leave a 0-byte file."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    assert not (tmp_path / "metrics.jsonl").exists()

    cb.on_log(_FakeArgs(), _FakeState(global_step=10, epoch=0.05),
              _FakeControl(), logs={"loss": 1.0, "learning_rate": 2e-4})
    assert (tmp_path / "metrics.jsonl").exists()
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())


def test_callback_appends_one_line_per_log(tmp_path: Path) -> None:
    """Three log calls → three JSONL lines, in order."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    for step, loss in [(10, 2.0), (20, 1.5), (30, 1.0)]:
        cb.on_log(_FakeArgs(), _FakeState(global_step=step, epoch=step / 100.0),
                  _FakeControl(), logs={"loss": loss, "learning_rate": 2e-4})
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())

    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(rows) == 3
    assert [r["step"] for r in rows] == [10, 20, 30]
    assert [r["loss"] for r in rows] == [2.0, 1.5, 1.0]
    assert all(r["kind"] == "train" for r in rows)


def test_callback_carries_step_and_epoch_metadata(tmp_path: Path) -> None:
    """Every line must carry step/epoch/kind so a downstream pandas
    consumer can plot a curve without re-deriving them."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    cb.on_log(_FakeArgs(), _FakeState(global_step=42, epoch=1.337),
              _FakeControl(), logs={"loss": 0.5, "learning_rate": 1e-4})
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())

    row = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["step"] == 42
    assert row["epoch"] == pytest.approx(1.337)
    assert row["kind"] == "train"


def test_callback_preserves_reg_kl_l2_keys(tmp_path: Path) -> None:
    """ModalityAwareSFTTrainer.log injects ``reg_kl`` / ``reg_l2`` into
    the logs dict. The JSONL must carry them verbatim — that's the
    point of the curve file for regularized runs."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    cb.on_log(
        _FakeArgs(), _FakeState(global_step=100, epoch=0.5), _FakeControl(),
        logs={"loss": 1.0, "learning_rate": 2e-4,
              "reg_kl": 0.012, "reg_l2": 0.003},
    )
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())
    row = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["reg_kl"] == pytest.approx(0.012)
    assert row["reg_l2"] == pytest.approx(0.003)


def test_callback_handles_eval_log_with_multi_val_keys(tmp_path: Path) -> None:
    """Mid-training multi-eval log → ``kind=="eval"`` and every
    ``eval_<key>_loss`` survives the round-trip."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    cb.on_log(
        _FakeArgs(), _FakeState(global_step=1000, epoch=2.0), _FakeControl(),
        logs={"eval_plant_loss": 0.8, "eval_nonplant_loss": 1.2,
              "eval_negative_loss": 0.4, "eval_offline_qa_loss": 0.9},
    )
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())
    row = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["kind"] == "eval"
    assert row["eval_plant_loss"] == pytest.approx(0.8)
    assert row["eval_offline_qa_loss"] == pytest.approx(0.9)


def test_callback_appends_across_resumes(tmp_path: Path) -> None:
    """Resume-from-checkpoint should APPEND to the existing
    metrics.jsonl, not truncate it. We simulate this by constructing
    two callback instances pointing at the same dir (matches what
    happens when finetune.py is restarted with --resume_from_checkpoint)."""
    cb1 = JsonlMetricsCallback(output_dir=tmp_path)
    cb1.on_log(_FakeArgs(), _FakeState(global_step=10, epoch=0.05),
               _FakeControl(), logs={"loss": 2.0, "learning_rate": 2e-4})
    cb1.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())

    cb2 = JsonlMetricsCallback(output_dir=tmp_path)
    cb2.on_log(_FakeArgs(), _FakeState(global_step=20, epoch=0.10),
               _FakeControl(), logs={"loss": 1.5, "learning_rate": 2e-4})
    cb2.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())

    rows = _read_jsonl(tmp_path / "metrics.jsonl")
    assert len(rows) == 2
    assert [r["step"] for r in rows] == [10, 20]


def test_callback_coerces_non_json_value_to_repr(tmp_path: Path) -> None:
    """If HF ever passes a non-JSON-serialisable value, the line must
    still parse — we degrade to repr rather than crash the training
    loop on a logging side-effect."""

    class _NotJsonable:
        def __repr__(self) -> str:
            return "<custom>"

    cb = JsonlMetricsCallback(output_dir=tmp_path)
    cb.on_log(_FakeArgs(), _FakeState(global_step=1, epoch=0.0),
              _FakeControl(), logs={"loss": 1.0, "weird": _NotJsonable()})
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())
    row = _read_jsonl(tmp_path / "metrics.jsonl")[0]
    assert row["loss"] == 1.0
    assert row["weird"] == "<custom>"


def test_callback_no_log_means_no_file(tmp_path: Path) -> None:
    """A callback instantiated for a dry-run code path that never sees
    an on_log call must not leave a phantom metrics.jsonl behind."""
    cb = JsonlMetricsCallback(output_dir=tmp_path)
    cb.on_log(_FakeArgs(), _FakeState(), _FakeControl(), logs=None)
    cb.on_train_end(_FakeArgs(), _FakeState(), _FakeControl())
    assert not (tmp_path / "metrics.jsonl").exists()
