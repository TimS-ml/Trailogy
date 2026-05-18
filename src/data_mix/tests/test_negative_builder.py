from __future__ import annotations

from pathlib import Path

import pytest

from data_mix.src.negative_builder import (
    NEGATIVE_PROMPT,
    NEGATIVE_RESPONSE,
    build_negative_records,
)
from data_mix.src.schema import validate_record


def test_constants():
    assert "plant" in NEGATIVE_PROMPT.lower()
    assert "don't" in NEGATIVE_RESPONSE.lower() or "do not" in NEGATIVE_RESPONSE.lower()


def test_build_negative_records_one_per_path():
    paths = [Path(f"/abs/img{i}.jpg") for i in range(5)]
    out = build_negative_records(paths)
    assert len(out) == 5
    for rec, p in zip(out, paths):
        validate_record(rec)
        assert rec["source"] == "negative"
        assert rec["image"] == str(p)
        assert rec["conversations"][0]["content"] == NEGATIVE_PROMPT
        assert rec["conversations"][1]["content"] == NEGATIVE_RESPONSE


def test_empty_input_returns_empty():
    assert build_negative_records([]) == []
