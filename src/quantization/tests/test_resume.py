"""Unit tests for the resume-after-crash state machine.

Covers every status transition + the crash-recovery scenario where a
process is killed mid-stage (state.json has ``in_progress``) and the
next launch must downgrade that to ``pending`` so the stage is rerun.

Run with::

    pytest quantization/tests/test_resume.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.resume import (  # noqa: E402
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_IN_PROGRESS,
    STATUS_PENDING,
    StateMachine,
)

STAGES = ("convert", "size", "smoke", "ppl")


def _make_sm(tmp_path: Path, variant: str = "test-variant") -> StateMachine:
    return StateMachine(
        state_path=tmp_path / "state.json",
        variant=variant,
        stages=STAGES,
    )


# ---------------------------------------------------------------------------
# load_or_init: cold start + warm start
# ---------------------------------------------------------------------------


def test_cold_init_creates_pending_state(tmp_path):
    sm = _make_sm(tmp_path)
    state = sm.load_or_init()
    assert state["variant"] == "test-variant"
    for stage in STAGES:
        assert state["stages"][stage]["status"] == STATUS_PENDING
        assert state["stages"][stage]["started_at"] is None
        assert state["stages"][stage]["finished_at"] is None
    # File on disk has the same content.
    on_disk = json.loads((tmp_path / "state.json").read_text())
    assert on_disk == state


def test_warm_init_preserves_done_stages(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("convert")
    sm.mark_done("convert", result={"foo": "bar"})

    # Simulate a separate process: brand new StateMachine on same file.
    sm2 = _make_sm(tmp_path)
    sm2.load_or_init()
    assert sm2.is_done("convert")
    assert sm2.state["stages"]["convert"]["result"] == {"foo": "bar"}
    assert sm2.status_of("size") == STATUS_PENDING


# ---------------------------------------------------------------------------
# Crash-recovery: in_progress → pending on next load
# ---------------------------------------------------------------------------


def test_in_progress_is_downgraded_to_pending_on_reload(tmp_path):
    """Simulates: stage started, process killed before mark_done, restart."""
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("convert")
    # ↑ At this point disk state has convert.status == in_progress.

    # New process starts — its load_or_init must recover.
    sm2 = _make_sm(tmp_path)
    sm2.load_or_init()
    assert sm2.status_of("convert") == STATUS_PENDING
    assert "stale_recovered_at" in sm2.state["stages"]["convert"]


def test_in_progress_recovery_does_not_clobber_other_done_stages(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("convert"); sm.mark_done("convert")
    sm.mark_in_progress("size"); sm.mark_done("size")
    sm.mark_in_progress("smoke")  # crash here

    sm2 = _make_sm(tmp_path)
    sm2.load_or_init()
    assert sm2.is_done("convert")
    assert sm2.is_done("size")
    assert sm2.status_of("smoke") == STATUS_PENDING
    assert sm2.status_of("ppl") == STATUS_PENDING


# ---------------------------------------------------------------------------
# Transitions: done / failed / reset
# ---------------------------------------------------------------------------


def test_mark_done_stores_result(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("ppl")
    sm.mark_done("ppl", result={"perplexity": 1234.5})
    assert sm.is_done("ppl")
    assert sm.state["stages"]["ppl"]["result"]["perplexity"] == pytest.approx(1234.5)
    assert sm.state["stages"]["ppl"]["error"] is None


def test_mark_failed_records_error_message(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("convert")
    sm.mark_failed("convert", error="boom: NotImplementedError")
    assert sm.status_of("convert") == STATUS_FAILED
    assert "boom" in sm.state["stages"]["convert"]["error"]


def test_reset_stage_brings_done_back_to_pending(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("smoke"); sm.mark_done("smoke", result={"x": 1})
    sm.reset_stage("smoke")
    assert sm.status_of("smoke") == STATUS_PENDING
    assert sm.state["stages"]["smoke"]["result"] is None


def test_reset_stage_brings_failed_back_to_pending(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    sm.mark_in_progress("smoke"); sm.mark_failed("smoke", error="x")
    sm.reset_stage("smoke")
    assert sm.status_of("smoke") == STATUS_PENDING


# ---------------------------------------------------------------------------
# Atomic persist: state.json never corrupted mid-write
# ---------------------------------------------------------------------------


def test_atomic_persist_uses_tmp_then_rename(tmp_path, monkeypatch):
    """os.replace must be called with a .tmp source — verify."""
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    seen = []
    real_replace = __import__("os").replace

    def spy(src, dst):
        seen.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr("os.replace", spy)
    sm.mark_in_progress("convert")
    sm.mark_done("convert")
    assert seen, "os.replace was never called"
    for src, dst in seen:
        assert str(src).endswith(".tmp"), f"non-atomic write: src={src}"
        assert str(dst) == str(tmp_path / "state.json")


# ---------------------------------------------------------------------------
# Misc guards
# ---------------------------------------------------------------------------


def test_methods_require_load_first(tmp_path):
    sm = _make_sm(tmp_path)
    with pytest.raises(RuntimeError, match="load_or_init"):
        sm.is_done("convert")


def test_unknown_stage_raises(tmp_path):
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    with pytest.raises(KeyError, match="bogus"):
        sm.mark_in_progress("bogus")


def test_schema_migration_adds_new_stages(tmp_path):
    """If state.json was written by an older runner with fewer stages,
    load_or_init must fill in the missing ones as pending."""
    legacy = {
        "variant": "x",
        "created_at": "2026-01-01T00:00:00Z",
        "stages": {
            # Only one stage from an older version of the pipeline.
            "convert": {
                "status": STATUS_DONE,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:01:00Z",
                "error": None,
                "result": None,
            },
        },
    }
    (tmp_path / "state.json").write_text(json.dumps(legacy))
    sm = _make_sm(tmp_path)
    sm.load_or_init()
    assert sm.is_done("convert")
    for new_stage in ("size", "smoke", "ppl"):
        assert sm.status_of(new_stage) == STATUS_PENDING
