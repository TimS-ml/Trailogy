"""Calibration-data builders shared across PTQ methods.

Any calibration-driven PTQ (GPTQ, AWQ, dynamic_quant) needs a small
dataset to estimate per-layer activation statistics. The data design
is route-agnostic — the same loaders feed B.1 (HF GPTQModel on CUDA)
and B.2 (mlx-lm core via the hybrid flow). See the team's calibration
data-design notes for the per-source ablation matrix.

Two loaders today:

- ``load_plantnet_calibration(...)``: domain text from PlantNet
  ``train.jsonl`` (eval/test/canonical leakage rejected).
- ``load_text_calibration(...)``: generic-English text from
  WikiText-103.

Both return ``list[{"text": str}]`` records consumable by
``gptqmodel.BaseQModel.quantize(calibration=...)``.

Eval-leak guard: ``CalibrationDataLeakError`` is raised if the caller
hands an eval/test/canonical-species file or an ``overfit100`` file.
Calibrating on those trivially preserves eval metrics while learning
nothing about generalization.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)


def _flatten_conversation_to_text(conv: list[dict]) -> str:
    """Fallback formatter when no tokenizer is supplied.

    Plain role-prefixed text — no fake chat-template markers. The point
    is that **this branch is calibration-distribution-incorrect by
    construction**; callers should provide a real ``tokenizer`` so
    ``apply_chat_template`` runs the same code path inference uses.
    The fallback exists for unit tests and for the rare case where
    a tokenizer cannot be loaded.
    """
    parts: list[str] = []
    for msg in conv:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(f"{role}: {block.get('text', '')}")
    return "\n".join(parts)


def _format_with_tokenizer(tokenizer: Any, conv: list[dict]) -> str | None:
    """Try ``tokenizer.apply_chat_template`` on the conv. Returns the
    formatted string, or ``None`` if the tokenizer rejects this conv
    shape (e.g. image-content blocks the text-only tokenizer doesn't
    handle — caller can fall back to plain flatten).
    """
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return None
    # Strip image-only blocks; keep text. Chat templates for text
    # tokenizers reject ``{"type": "image"}`` blocks.
    cleaned: list[dict] = []
    for msg in conv:
        content = msg.get("content", "")
        if isinstance(content, list):
            text_blocks = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            if not text_blocks:
                continue
            cleaned.append({"role": msg.get("role", "user"), "content": "\n".join(text_blocks)})
        elif isinstance(content, str) and content:
            cleaned.append({"role": msg.get("role", "user"), "content": content})
    if not cleaned:
        return None
    try:
        return tokenizer.apply_chat_template(
            cleaned, tokenize=False, add_generation_prompt=False
        )
    except Exception as e:  # noqa: BLE001
        log.warning("apply_chat_template failed (%s); falling back to flatten.", e)
        return None


class CalibrationDataLeakError(RuntimeError):
    """Raised when a calibration source would leak into eval data.

    The failure modes we guard against:

    1. Passing eval/test/canonical-species files as the calibration
       source. PTQ would then be solving for activations on the exact
       same distribution the eval scores. Results would look amazing
       and mean nothing.
    2. Passing an ``overfit100`` split (where train == eval by design;
       used for memorization-ceiling tests only). Every quant variant
       would score near-100 % and tell us nothing about real
       generalization.
    """


def reject_calibration_leak(plantnet_jsonl: Path) -> None:
    """Hard-fail if the calibration path looks like a known eval/overfit set.

    Public entry point — callers in any PTQ method should invoke this
    before sampling any record from ``plantnet_jsonl``.
    """
    s = str(plantnet_jsonl).lower()
    name = plantnet_jsonl.name.lower()
    stem = plantnet_jsonl.stem.lower()
    eval_names = {"val", "eval", "test"}
    split = next(
        (
            split
            for split in eval_names
            if name == f"{split}.jsonl"
            or stem == split
            or stem.endswith(f"-{split}")
            or stem.endswith(f"_{split}")
            or any(
                part.lower() == split or part.lower().startswith(f"{split}.")
                for part in plantnet_jsonl.parts
            )
        ),
        None,
    )
    if split is not None:
        raise CalibrationDataLeakError(
            f"Refusing to use {plantnet_jsonl} for calibration. "
            f"{split} is an eval/test split — use train.jsonl as the calibration "
            "source. If you really want to override, rename the file."
        )
    if "canonical" in s and "species" in s:
        raise CalibrationDataLeakError(
            f"Refusing to use {plantnet_jsonl} for calibration. "
            "canonical-species data is reserved for evaluation/analysis, "
            "not PTQ calibration."
        )
    if "overfit100" in s or "overfit-100" in s:
        raise CalibrationDataLeakError(
            f"Refusing to use {plantnet_jsonl} for calibration. "
            "overfit100 data is train==eval by design (memorization-ceiling "
            "test). Calibrating on it will trivially appear to preserve "
            "all eval metrics. Use plantnet-50k train.jsonl."
        )


# Back-compat alias for code that used the private name.
_reject_calibration_leak = reject_calibration_leak


def load_plantnet_calibration(
    plantnet_jsonl: Path | str,
    n_samples: int,
    seed: int,
    tokenizer: Any | None = None,
) -> list[dict]:
    """Load N PlantNet train samples as calibration records.

    Calibration MUST come from train.jsonl (or another held-out split
    disjoint from eval). Eval-set leakage is rejected by
    ``reject_calibration_leak``.

    Returns ``list[{"text": str}]``. Image-content blocks are stripped
    (text-only calibration; the vision tower is not GPTQ-targeted in
    the language-projection direction).

    If ``tokenizer`` is supplied and exposes ``apply_chat_template``,
    each conversation is formatted via the real chat template so the
    calibration-time activation distribution matches inference. Without
    a tokenizer the function falls back to a plain ``role: content``
    flatten — sufficient for unit tests but **not** calibration-
    distribution-correct.
    """
    plantnet_jsonl = Path(plantnet_jsonl)
    reject_calibration_leak(plantnet_jsonl)
    if not plantnet_jsonl.is_file():
        log.warning(
            "PlantNet JSONL not found at %s — skipping that calib source.",
            plantnet_jsonl,
        )
        return []
    records: list[dict] = []
    with plantnet_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    rng = random.Random(seed)
    rng.shuffle(records)
    records = records[:n_samples]

    out: list[dict] = []
    used_template = 0
    used_flatten = 0
    for rec in records:
        conv = rec.get("conversations") or rec.get("messages") or []
        if not conv:
            continue
        text = _format_with_tokenizer(tokenizer, conv)
        if text is not None:
            used_template += 1
        else:
            text = _flatten_conversation_to_text(conv)
            used_flatten += 1
        if text:
            out.append({"text": text})
    log.info(
        "PlantNet calibration: %d records assembled (chat-template=%d, flatten=%d).",
        len(out), used_template, used_flatten,
    )
    return out


def load_text_calibration(n_samples: int, seed: int, seq_len: int) -> list[dict]:
    """Load N WikiText-103 segments as generic-language calibration."""
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("`datasets` not installed — skipping text calibration.")
        return []
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    rng = random.Random(seed + 1)  # offset from PlantNet seed
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    out: list[dict] = []
    char_budget = seq_len * 4  # rough chars-per-token for English
    for i in idx:
        text = ds[int(i)].get("text", "").strip()
        if len(text) < 200:
            continue
        out.append({"text": text[:char_budget]})
        if len(out) >= n_samples:
            break
    log.info("Text calibration: %d records assembled.", len(out))
    return out


def build_calibration_dataset(
    n_calib_plantnet: int,
    n_calib_text: int,
    calib_seq_len: int,
    calib_seed: int,
    plantnet_jsonl: Path | str | None,
    tokenizer: Any | None = None,
) -> list[dict]:
    """Concatenate domain (PlantNet) and general-text (WikiText) calibration.

    Returns the shuffled union. Each input source can be disabled by
    passing ``n_calib_* = 0`` (or ``plantnet_jsonl=None`` for the
    PlantNet half). ``tokenizer`` is forwarded to PlantNet loader so
    the chat template formats samples to match inference.
    """
    samples: list[dict] = []
    if plantnet_jsonl is not None and n_calib_plantnet > 0:
        samples.extend(
            load_plantnet_calibration(
                plantnet_jsonl, n_calib_plantnet, calib_seed, tokenizer=tokenizer
            )
        )
    if n_calib_text > 0:
        samples.extend(
            load_text_calibration(n_calib_text, calib_seed, calib_seq_len)
        )
    rng = random.Random(calib_seed)
    rng.shuffle(samples)
    log.info("Total calibration samples: %d", len(samples))
    return samples
