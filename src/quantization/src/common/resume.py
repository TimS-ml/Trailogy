"""Resume-after-crash state machine for staged pipelines.

A ``StateMachine`` owns one ``state.json`` file describing per-stage
status for a single pipeline variant. Used by
``scripts.run.mac_mlx_lm`` to persist progress across
runs so a Ctrl-C / kill / power loss / terminal-timeout interruption
resumes at the next stage on re-launch.

Status lifecycle for each stage::

    pending  ──mark_in_progress──▶  in_progress
    in_progress  ──mark_done──▶  done
    in_progress  ──mark_failed──▶  failed
    (next load) in_progress  ──load_or_init──▶  pending  (stale lock)
    done  ──reset_stage──▶  pending  (forced re-run)
    failed  ──reset_stage──▶  pending  (retry)

On ``load_or_init`` any stage left in ``in_progress`` is downgraded to
``pending`` — the previous process didn't reach ``mark_done`` or
``mark_failed``, so we assume it was killed and the stage needs to
re-run. Stages MUST therefore be idempotent.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

STATUS_PENDING = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# A "stage" is just a callable; declared here as a TypeAlias so the
# runner can type-hint its registry dict.
Stage = Callable[..., Any]


class StageOutcome(dict):
    """Convenience alias — stages return any JSON-serializable dict."""


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0, tzinfo=None).isoformat() + "Z"


class StateMachine:
    """Owns ``state.json`` for one variant.

    Construction is cheap and side-effect-free. Call ``load_or_init``
    before any other method.
    """

    def __init__(
        self,
        state_path: Path,
        variant: str,
        stages: Iterable[str],
    ) -> None:
        self.state_path = Path(state_path)
        self.variant = variant
        self.stages = tuple(stages)
        self._state: dict[str, Any] | None = None  # populated by load_or_init

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_or_init(self) -> dict[str, Any]:
        if self.state_path.exists():
            self._state = json.loads(self.state_path.read_text())
            # Schema migration / fill any newly-added stages.
            for st in self.stages:
                self._state["stages"].setdefault(st, self._fresh_stage())
            # Downgrade stale in_progress to pending.
            for st, rec in self._state["stages"].items():
                if rec.get("status") == STATUS_IN_PROGRESS:
                    rec["status"] = STATUS_PENDING
                    rec["stale_recovered_at"] = _utcnow_iso()
            self._persist()
        else:
            self._state = {
                "variant": self.variant,
                "created_at": _utcnow_iso(),
                "stages": {st: self._fresh_stage() for st in self.stages},
            }
            self._persist()
        return self._state

    @staticmethod
    def _fresh_stage() -> dict[str, Any]:
        return {
            "status": STATUS_PENDING,
            "started_at": None,
            "finished_at": None,
            "error": None,
            "result": None,
        }

    def _persist(self) -> None:
        assert self._state is not None
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.state_path)  # atomic on POSIX

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_done(self, stage: str) -> bool:
        self._require_loaded()
        return self._state["stages"][stage]["status"] == STATUS_DONE  # type: ignore[index]

    def status_of(self, stage: str) -> str:
        self._require_loaded()
        return self._state["stages"][stage]["status"]  # type: ignore[index]

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    def mark_in_progress(self, stage: str) -> None:
        self._require_stage(stage)
        rec = self._state["stages"][stage]  # type: ignore[index]
        rec["status"] = STATUS_IN_PROGRESS
        rec["started_at"] = _utcnow_iso()
        rec["pid"] = os.getpid()
        rec["error"] = None
        self._persist()

    def mark_done(self, stage: str, result: Any | None = None) -> None:
        self._require_stage(stage)
        rec = self._state["stages"][stage]  # type: ignore[index]
        rec["status"] = STATUS_DONE
        rec["finished_at"] = _utcnow_iso()
        rec["result"] = result
        rec["error"] = None
        self._persist()

    def mark_failed(self, stage: str, error: str) -> None:
        self._require_stage(stage)
        rec = self._state["stages"][stage]  # type: ignore[index]
        rec["status"] = STATUS_FAILED
        rec["finished_at"] = _utcnow_iso()
        rec["error"] = error
        self._persist()

    def reset_stage(self, stage: str) -> None:
        """Force a stage back to ``pending`` (and drop any cached result)."""
        self._require_stage(stage)
        self._state["stages"][stage] = self._fresh_stage()  # type: ignore[index]
        self._persist()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        if self._state is None:
            raise RuntimeError("StateMachine.load_or_init() must be called first")

    def _require_stage(self, stage: str) -> None:
        self._require_loaded()
        if stage not in self._state["stages"]:  # type: ignore[index]
            raise KeyError(f"Unknown stage '{stage}'. Known: {sorted(self.stages)}")

    @property
    def state(self) -> dict[str, Any]:
        self._require_loaded()
        return self._state  # type: ignore[return-value]
