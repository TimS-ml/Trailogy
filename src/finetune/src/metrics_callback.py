"""JSONL metrics callback for HF Trainer.

Streams every ``trainer.log()`` payload as one JSON line to
``{output_dir}/metrics.jsonl``. This complements HF's own
``trainer_state.json`` (which only lands on disk at save_steps and
buries logs inside other Trainer state) and any external trackers
(wandb, tensorboard) wired via ``report_to``.

Why a separate file:
  * **Crash-safe**: each log line is fsync'd to disk on emit, so a
    mid-training jetsam / OOM still leaves the curve up to the last
    successful log step.
  * **First-class loss curve**: a single ``metrics.jsonl`` per run is
    trivially loaded with ``pandas.read_json("metrics.jsonl",
    lines=True)`` for plotting / cross-run comparison; no need to
    parse ``trainer_state.json`` from each checkpoint dir.
  * **No external dependency**: works on offline boxes / CI containers
    without wandb / tensorboard. Stacked with ``report_to: wandb`` when
    online, gives both real-time dashboard AND local source-of-truth.

Each emitted line carries:
  * ``step``  — ``state.global_step`` at emit time
  * ``epoch`` — ``state.epoch`` (float; HF emits this even mid-step)
  * ``kind``  — ``"train"`` if any ``eval_*`` key is absent and ``loss``
                 is present; ``"eval"`` if any key starts with ``eval_``;
                 ``"other"`` otherwise (e.g. final training summary).
  * the full ``logs`` dict, flattened (no key rewriting).

The callback is intentionally additive: it never modifies the logs
dict, so wandb / tensorboard / stdout see the exact same payload they
would without the callback wired in.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover - transformers is a hard project dep
    TrainerCallback = object  # type: ignore[misc,assignment]

log = logging.getLogger(__name__)


def _classify_log(logs: Dict[str, Any]) -> str:
    """Tag the log payload with a coarse kind for downstream filtering.

    HF Trainer emits three flavours of log dicts through the same
    ``on_log`` callback:

      1. Train step logs: ``{"loss": ..., "learning_rate": ...,
         "grad_norm": ..., "epoch": ...}`` (plus our ``reg_kl`` /
         ``reg_l2`` injections).
      2. Eval logs: ``{"eval_<key>_loss": ..., "eval_<key>_runtime": ...,
         ...}`` — when a multi-eval-dataset dict is passed, HF prefixes
         every eval metric with the dataset key.
      3. End-of-training summary: ``{"train_runtime": ...,
         "train_samples_per_second": ..., ...}``.

    The classification is best-effort and stable — downstream code can
    filter ``kind == "train"`` for the loss curve, ``kind == "eval"``
    for the multi-val-set curve, and ignore ``"other"``.
    """
    has_eval = any(k.startswith("eval_") for k in logs.keys())
    if has_eval:
        return "eval"
    if "loss" in logs and "learning_rate" in logs:
        return "train"
    return "other"


class JsonlMetricsCallback(TrainerCallback):
    """HF TrainerCallback that mirrors every ``on_log`` to a JSONL file.

    Parameters
    ----------
    output_dir:
        Directory the file is written into. Always ``cfg.training.output_dir``
        in practice. The file is named ``metrics.jsonl`` inside it.
    filename:
        Override the default ``"metrics.jsonl"`` if needed (e.g. for
        ablation runs that share an output_dir).
    flush_each_line:
        When True (default), flushes + ``os.fsync()`` after every line so
        a kernel jetsam / hard kill loses at most the in-flight log.
        Set False only if you measure logging overhead and the run is
        emitting thousands of logs/sec (we don't).

    The file handle is opened lazily on the first ``on_log`` so that a
    callback constructed in a dry-run / no-train code path doesn't
    create an empty ``metrics.jsonl``.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        filename: str = "metrics.jsonl",
        flush_each_line: bool = True,
    ) -> None:
        self._dir = Path(output_dir)
        self._filename = filename
        self._flush_each_line = flush_each_line
        self._path: Optional[Path] = None
        self._fh: Optional[Any] = None

    # ----- lifecycle -------------------------------------------------

    def _ensure_open(self) -> None:
        if self._fh is not None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._filename
        # Append mode so resume-from-checkpoint runs don't clobber the
        # pre-resume curve. The reader can dedupe on ``step`` if needed,
        # but in practice resumes continue past the last logged step so
        # there's no overlap.
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        log.info("JsonlMetricsCallback: appending metrics to %s", self._path)

    def _close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):
                # fh might already be closed; this is best-effort shutdown.
                pass
            try:
                self._fh.close()
            except (OSError, ValueError):
                pass
            self._fh = None

    def __del__(self) -> None:  # best-effort, GC may skip
        self._close()

    # ----- HF TrainerCallback hooks ---------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[override]
        """Append one JSON line per HF Trainer log emit.

        We deliberately do NOT mutate ``logs`` — it's the same dict
        wandb / tensorboard / stdout see, and the trainer expects it
        untouched.
        """
        if logs is None:
            return control
        self._ensure_open()
        record: Dict[str, Any] = {
            "step": int(getattr(state, "global_step", 0) or 0),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "kind": _classify_log(logs),
        }
        # Copy every numeric / scalar key from logs verbatim. Non-JSON-
        # serialisable values (rare; HF emits floats/ints/strs) are
        # coerced to ``repr`` so the line still parses.
        for k, v in logs.items():
            try:
                json.dumps(v)
                record[k] = v
            except (TypeError, ValueError):
                record[k] = repr(v)
        assert self._fh is not None  # _ensure_open guarantees this
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if self._flush_each_line:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except (OSError, ValueError):
                # Disk full / closed handle — drop quietly so training
                # itself doesn't crash on a logging side-effect.
                pass
        return control

    def on_train_end(self, args, state, control, **kwargs):  # type: ignore[override]
        """Close the file at train end so the OS releases the handle."""
        self._close()
        return control
