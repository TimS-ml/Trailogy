"""End-to-end resume smoke test for ``run_mac_mlx_lm``.

Uses a fake stage table so we don't depend on the real mlx_lm
forward pass. Verifies:

1. Cold run executes all stages in order.
2. Re-run with the same args skips already-done stages.
3. Crashed stage (raises) is marked failed; next run re-attempts it.
4. ``--force-stage`` clears the done flag and re-runs.
5. ``--only-stage`` runs exactly one stage, skips the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.resume import (  # noqa: E402
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
)
from scripts.run import mac_mlx_lm as runner  # noqa: E402

CONFIG_YAML = """\
method: gptq
quant:
  bits: 4
"""


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "fake.yaml"
    p.write_text(CONFIG_YAML)
    return p


@pytest.fixture
def fake_stages(monkeypatch, tmp_path):
    """Replace STAGE_FUNCS with deterministic fakes that write a marker file."""

    log: list[str] = []

    def make_stage(name: str, *, raises: bool = False):
        def fn(cfg, input_dir, output_dir):
            log.append(name)
            if raises:
                raise RuntimeError(f"{name} fake-crash")
            (output_dir / f"{name}.marker").write_text("ok")
            return {"stage": name, "n_log": len(log)}
        return fn

    fakes = {
        "convert": make_stage("convert"),
        "size": make_stage("size"),
        "smoke": make_stage("smoke"),
        "ppl": make_stage("ppl"),
    }
    monkeypatch.setattr(runner, "STAGE_FUNCS", fakes)
    return {"log": log, "fakes": fakes, "make_stage": make_stage}


def _run(tmp_path: Path, cfg: Path, *extra) -> int:
    return runner.main([
        "--variant", "test-e2e",
        "--config", str(cfg),
        "--input-dir", str(tmp_path / "in"),
        "--output-dir", str(tmp_path / "out"),
        *extra,
    ])


# ---------------------------------------------------------------------------


def test_cold_run_executes_all_stages_in_order(tmp_path, fake_stages):
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()
    rc = _run(tmp_path, cfg)
    assert rc == 0
    assert fake_stages["log"] == ["convert", "size", "smoke", "ppl"]
    for s in runner.STAGES:
        assert (tmp_path / "out" / f"{s}.marker").exists()


def test_rerun_skips_done_stages(tmp_path, fake_stages):
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()
    _run(tmp_path, cfg)
    fake_stages["log"].clear()

    rc = _run(tmp_path, cfg)
    assert rc == 0
    assert fake_stages["log"] == []  # nothing re-ran


def test_failed_stage_is_retried_on_next_run(tmp_path, fake_stages, monkeypatch):
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()
    # First run: smoke crashes.
    fakes = fake_stages["fakes"]
    fakes["smoke"] = fake_stages["make_stage"]("smoke", raises=True)
    monkeypatch.setattr(runner, "STAGE_FUNCS", fakes)

    rc = _run(tmp_path, cfg)
    assert rc == 1  # runner returns 1 on failure
    assert "smoke" in fake_stages["log"]
    assert not (tmp_path / "out" / "smoke.marker").exists()

    # Second run: smoke no longer crashes.
    fake_stages["log"].clear()
    fakes["smoke"] = fake_stages["make_stage"]("smoke")
    monkeypatch.setattr(runner, "STAGE_FUNCS", fakes)

    rc = _run(tmp_path, cfg)
    assert rc == 0
    # convert + size already done; smoke retries, ppl runs.
    assert fake_stages["log"] == ["smoke", "ppl"]


def test_force_stage_reruns_completed_stage(tmp_path, fake_stages):
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()
    _run(tmp_path, cfg)
    fake_stages["log"].clear()

    rc = _run(tmp_path, cfg, "--force-stage", "size")
    assert rc == 0
    assert fake_stages["log"] == ["size"]  # ONLY size re-ran (others already done)


def test_only_stage_runs_just_that_stage(tmp_path, fake_stages):
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()
    rc = _run(tmp_path, cfg, "--only-stage", "size")
    assert rc == 0
    assert fake_stages["log"] == ["size"]
    # State file shows size done, others still pending.
    import json
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    assert state["stages"]["size"]["status"] == STATUS_DONE
    assert state["stages"]["convert"]["status"] == STATUS_PENDING
    assert state["stages"]["ppl"]["status"] == STATUS_PENDING


def test_crashed_in_progress_recovers_on_rerun(tmp_path, fake_stages, monkeypatch):
    """Hard kill mid-stage: state.json left at in_progress.
    Next launch downgrades to pending and the stage re-runs cleanly."""
    cfg = _write_cfg(tmp_path)
    (tmp_path / "in").mkdir()

    # Manually create an in_progress state to simulate a hard kill
    # between mark_in_progress and the stage's natural exit.
    from src.common.resume import StateMachine
    sm = StateMachine(
        state_path=tmp_path / "out" / "state.json",
        variant="test-e2e",
        stages=runner.STAGES,
    )
    (tmp_path / "out").mkdir()
    sm.load_or_init()
    sm.mark_in_progress("convert")
    # ↑ simulates a killed process

    rc = _run(tmp_path, cfg)
    assert rc == 0
    # All stages ran (convert was recovered from stale in_progress).
    assert fake_stages["log"] == ["convert", "size", "smoke", "ppl"]
