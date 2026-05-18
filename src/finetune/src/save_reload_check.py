"""Save -> reload tensor preservation utilities.

Purpose
-------
Detect the AGENTS.md "orphan tensors" bug class:

    The adapter was trained on an older transformers that had k_proj /
    v_proj as nn.Linear in all 35 layers. On reload, PEFT silently
    dropped the 80 "orphan" tensors. The in-memory model and the
    reloaded model produced different outputs, but no error fired.

This module provides pure-Python comparison primitives that the
in-pipeline tripwire (`finetune.py`) and the heavyweight GPU smoke
script (`scripts/inspect/save_reload.py`) both build on:

  * `extract_savable_state(peft_model)` — what PEFT would write to disk.
  * `load_adapter_state(adapter_dir)` — what is on disk.
  * `diff_state(a, b, atol, rtol)` — set + value diff with tolerances.
  * `diff_in_memory_vs_disk(...)` — convenience wrapper for the tripwire.
  * `assert_no_diff(diff, label)` — raise with actionable message.

All comparison functions are torch-aware but require no CUDA, so they
test on CPU in CI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


# (key, reason) — reason is a short human-readable string ("shape", "dtype",
# "bytes", "value (atol/rtol)") used in error messages.
ValueMismatch = Tuple[str, str]


@dataclass(frozen=True)
class StateDiff:
    """Symmetric difference of two state dicts.

    only_in_a / only_in_b are key sets — the "orphan tensor" bug from
    AGENTS.md surfaces as a non-empty only_in_a (in-memory had it, disk
    didn't) or only_in_b (disk has tensors the reloaded model lost).

    value_mismatched lists keys present in both but whose tensors differ
    by shape, dtype, or numeric content (subject to the configured atol /
    rtol when diffing).
    """

    only_in_a: Set[str]
    only_in_b: Set[str]
    value_mismatched: List[ValueMismatch] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.only_in_a
            and not self.only_in_b
            and not self.value_mismatched
        )


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------


def extract_savable_state(peft_model: Any) -> Dict[str, torch.Tensor]:
    """Return the state dict PEFT would write via save_pretrained.

    Uses peft.utils.save_and_load.get_peft_model_state_dict — the same
    function save_pretrained calls internally — so the key set + bytes
    we get here MUST match adapter_model.safetensors after save.

    Detached + cloned so callers can free the trainer without losing the
    snapshot.
    """
    try:  # newer PEFT
        from peft.utils.save_and_load import get_peft_model_state_dict
    except ImportError:  # older PEFT
        from peft import get_peft_model_state_dict  # type: ignore[no-redef]

    raw = get_peft_model_state_dict(peft_model)
    return {k: v.detach().clone().cpu() for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Disk loading
# ---------------------------------------------------------------------------


def _adapter_safetensors_files(adapter_dir: Path) -> List[Path]:
    """Return every safetensors file in an adapter directory, sorted.

    Handles both the single-file layout (adapter_model.safetensors) and
    the sharded layout (model-00001-of-NN.safetensors). The single-file
    name is the PEFT default; sharded shows up on large modules_to_save
    sets (e.g. r=256 + projector + vision layers).
    """
    candidates: List[Path] = []
    single = adapter_dir / "adapter_model.safetensors"
    if single.exists():
        candidates.append(single)
    # Sharded PEFT adapters keep the `adapter_model.safetensors.index.json`
    # alongside multiple `model-00001-of-NN.safetensors` files.
    candidates.extend(sorted(adapter_dir.glob("model-*-of-*.safetensors")))
    candidates.extend(
        p for p in sorted(adapter_dir.glob("adapter_model-*-of-*.safetensors"))
        if p not in candidates
    )
    return candidates


def load_adapter_state(adapter_dir: Path | str) -> Dict[str, torch.Tensor]:
    """Load every tensor in an adapter directory into a single dict.

    Raises FileNotFoundError if the directory has no safetensors files at
    all — the tripwire is meaningless without an adapter to inspect.
    """
    adapter_dir = Path(adapter_dir)
    files = _adapter_safetensors_files(adapter_dir)
    if not files:
        raise FileNotFoundError(
            f"No safetensors files found in {adapter_dir}. "
            "Expected adapter_model.safetensors or sharded "
            "model-*-of-*.safetensors."
        )

    from safetensors.torch import load_file

    state: Dict[str, torch.Tensor] = {}
    for f in files:
        for k, v in load_file(str(f)).items():
            if k in state:
                raise ValueError(
                    f"Tensor key {k!r} appears in multiple safetensors "
                    f"shards under {adapter_dir} — refusing to silently "
                    "overwrite. Adapter directory is corrupt."
                )
            state[k] = v
    return state


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def diff_state(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> StateDiff:
    """Compute set + value diff between two state dicts.

    Default atol/rtol = 0 means byte equality (after casting both to a
    common dtype for the compare). For numeric tolerance comparisons
    (e.g. bf16 roundtripped through fp32) pass non-zero atol / rtol.

    Note: dtype mismatch is always flagged regardless of tolerance —
    silently casting bf16 to fp32 on save would change the deployed
    quality even if values match within tolerance.
    """
    keys_a = set(a)
    keys_b = set(b)
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    value_mismatched: List[ValueMismatch] = []

    for key in sorted(keys_a & keys_b):
        ta = a[key]
        tb = b[key]
        if ta.shape != tb.shape:
            value_mismatched.append((key, f"shape ({tuple(ta.shape)} vs {tuple(tb.shape)})"))
            continue
        if ta.dtype != tb.dtype:
            value_mismatched.append((key, f"dtype ({ta.dtype} vs {tb.dtype})"))
            continue
        # Same shape + dtype. Use allclose with the configured tolerance.
        # On dtype=fp32 with atol=rtol=0 this reduces to bytewise equality.
        a_cmp = ta.detach().cpu()
        b_cmp = tb.detach().cpu()
        if atol == 0.0 and rtol == 0.0:
            equal = bool(torch.equal(a_cmp, b_cmp))
            reason = "bytes"
        else:
            equal = bool(
                torch.allclose(a_cmp.float(), b_cmp.float(), atol=atol, rtol=rtol)
            )
            reason = f"value (atol={atol}, rtol={rtol})"
        if not equal:
            value_mismatched.append((key, reason))

    return StateDiff(only_in_a=only_a, only_in_b=only_b, value_mismatched=value_mismatched)


def diff_in_memory_vs_disk(
    in_memory: Dict[str, torch.Tensor],
    adapter_dir: Path | str,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> StateDiff:
    """Compare an in-memory PEFT state dict against what was just saved.

    This is the invariant the in-pipeline tripwire enforces: after
    trainer.save_model, what's on disk must equal what get_peft_model_state_dict
    returns. Any drift is either a save-side bug (HF/PEFT corrupted the
    file) or a stale-adapter situation (the directory contains tensors
    from a previous run that PEFT didn't overwrite).
    """
    disk = load_adapter_state(adapter_dir)
    return diff_state(in_memory, disk, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------


_MAX_KEYS_IN_MESSAGE = 8


def _format_key_set(keys: Set[str]) -> str:
    if not keys:
        return "(none)"
    sample = sorted(keys)[:_MAX_KEYS_IN_MESSAGE]
    remainder = len(keys) - len(sample)
    out = ", ".join(sample)
    if remainder > 0:
        out += f", ... ({len(keys)} total, {remainder} more)"
    return out


def _format_value_mismatches(mismatches: List[ValueMismatch]) -> str:
    if not mismatches:
        return "(none)"
    sample = mismatches[:_MAX_KEYS_IN_MESSAGE]
    remainder = len(mismatches) - len(sample)
    out = "; ".join(f"{name} [{reason}]" for name, reason in sample)
    if remainder > 0:
        out += f"; ... ({len(mismatches)} total, {remainder} more)"
    return out


def assert_no_diff(diff: StateDiff, *, label: str) -> None:
    """Raise RuntimeError when diff is non-empty.

    Message intentionally name-drops "orphan tensors" so operators
    immediately recognise the AGENTS.md bug class.
    """
    if diff.is_empty():
        return

    parts: List[str] = [
        f"Adapter {label} mismatch — possible orphan-tensor regression "
        "(see AGENTS.md). At least one trainable tensor was dropped or "
        "corrupted between in-memory and on-disk / reloaded state.",
    ]
    if diff.only_in_a:
        parts.append(
            f"  Missing from second state (orphan in first): "
            f"{_format_key_set(diff.only_in_a)}"
        )
    if diff.only_in_b:
        parts.append(
            f"  Missing from first state (orphan in second): "
            f"{_format_key_set(diff.only_in_b)}"
        )
    if diff.value_mismatched:
        parts.append(
            f"  Value mismatched: {_format_value_mismatches(diff.value_mismatched)}"
        )
    raise RuntimeError("\n".join(parts))
