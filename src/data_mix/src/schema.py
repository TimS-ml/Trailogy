"""Unified record schema for the data_mix corpus.

Schema (v2):
    {
        "image": "<absolute path>" | None,   # None = text-only (smoltalk)
        "conversations": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
            ...  # strict user/assistant alternation, starting with user
        ],
        "source": "plant" | "na_plantae" | "cambrian" | "llava" | "smoltalk" | "negative" | "offline_qa"
    }

v2 changes vs v1:
- `image` accepts None (was: non-empty string only). None marks a
  text-only record that the trainer routes through a vision-skip batch
  via ModalityAwareBatchSampler. Empty string still rejected.
- `source` adds "llava" (HuggingFaceH4/llava-instruct-mix-vsft replaces
  Cambrian-10M for new mixes; "cambrian" retained for v1 mix-20k).
"""
from __future__ import annotations

from typing import Any, Mapping

ALLOWED_SOURCES = frozenset({
    "plant", "na_plantae", "cambrian", "llava",
    "smoltalk", "negative", "offline_qa",
})


class RecordError(ValueError):
    """Raised when a record does not match the unified schema."""


def validate_record(record: Mapping[str, Any]) -> None:
    if "image" not in record:
        raise RecordError("record missing 'image' key")
    img = record["image"]
    if img is not None and (not isinstance(img, str) or not img):
        raise RecordError(
            f"'image' must be None or a non-empty string, got {img!r}"
        )

    convs = record.get("conversations")
    if not isinstance(convs, list) or len(convs) < 2:
        raise RecordError(
            f"'conversations' must be a list of length >= 2, got {convs!r}"
        )

    expected_roles = ("user", "assistant")
    for i, turn in enumerate(convs):
        if not isinstance(turn, Mapping):
            raise RecordError(f"conversations[{i}] must be a mapping")
        role = turn.get("role")
        content = turn.get("content")
        if role != expected_roles[i % 2]:
            raise RecordError(
                f"conversations alternation broken at index {i}: "
                f"got role={role!r}, expected {expected_roles[i % 2]!r}"
            )
        if not isinstance(content, str) or not content:
            raise RecordError(
                f"conversations[{i}] 'content' must be non-empty string"
            )

    src = record.get("source")
    if src not in ALLOWED_SOURCES:
        raise RecordError(
            f"'source' must be one of {sorted(ALLOWED_SOURCES)}, got {src!r}"
        )
