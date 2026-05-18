"""Tests for ``data_mix/src/offline_qa_sampler.py``.

offline_qa is a small (~42 records) text-only persona-shaping corpus
sourced from ``hikeCompanion/assets/data_offline_qa/offline_qa.json``.
The file is a JSON list of ``{"question": str, "answer": str}`` pairs
designed to teach the model the "I'm an offline AI" persona so that
prompts like "are you ChatGPT?" / "search Google for me" get an
in-character refusal rather than the base Gemma response.

The sampler:
  - Reads the JSON file from disk.
  - Converts each entry to the v2 unified schema with
    ``image=None`` / ``source="offline_qa"``.
  - Holds out a small deterministic val slice (default 10 %, min 1).
  - Returns ``(train, val)`` lists ready to write to JSONL.

Tests cover schema, determinism, val carve-out semantics, and the
'no oversample' guarantee (we do NOT repeat entries — repeating would
teach the model the specific phrasings rather than the persona).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_mix.src.offline_qa_sampler import (
    load_offline_qa_records,
    sample_offline_qa_records,
)
from data_mix.src.schema import validate_record


# ---------------------------------------------------------------------------
# load_offline_qa_records — JSON → v2 schema
# ---------------------------------------------------------------------------


def _write_qa(path: Path, qa_list: list[dict]) -> None:
    path.write_text(json.dumps(qa_list))


def test_load_converts_to_v2_schema(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(
        p,
        [
            {"question": "Are you offline?", "answer": "Yes."},
            {"question": "Can you Google?", "answer": "No, I'm offline."},
        ],
    )
    recs = load_offline_qa_records(p)
    assert len(recs) == 2
    for rec in recs:
        validate_record(rec)
        assert rec["source"] == "offline_qa"
        assert rec["image"] is None
        assert rec["conversations"][0]["role"] == "user"
        assert rec["conversations"][1]["role"] == "assistant"
    # Q + A bytes survive intact.
    assert recs[0]["conversations"][0]["content"] == "Are you offline?"
    assert recs[0]["conversations"][1]["content"] == "Yes."


def test_load_rejects_missing_fields(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": "no answer"}])  # missing 'answer'
    with pytest.raises(ValueError):
        load_offline_qa_records(p)


def test_load_rejects_empty_answer(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": "Q", "answer": ""}])
    with pytest.raises(ValueError):
        load_offline_qa_records(p)


def test_load_rejects_non_list_root(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    p.write_text(json.dumps({"question": "Q", "answer": "A"}))
    with pytest.raises(ValueError):
        load_offline_qa_records(p)


def test_load_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_offline_qa_records(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# sample_offline_qa_records — train/val split + no-oversample guarantee
# ---------------------------------------------------------------------------


def test_sample_no_oversample_default(tmp_path: Path) -> None:
    """The corpus is tiny (~42 records); repeating entries to inflate the
    bucket would teach the model the exact phrasings rather than the
    persona. Default behaviour: each record appears at most once across
    train + val combined."""
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)])
    train, val = sample_offline_qa_records(p, val_ratio=0.1, seed=0)
    combined = train + val
    questions = [r["conversations"][0]["content"] for r in combined]
    assert len(questions) == len(set(questions)) == 10


def test_sample_val_ratio_carves_out_min_one(tmp_path: Path) -> None:
    """val_ratio=0.1 over 10 records → 1 val + 9 train. Even at tiny
    corpus sizes we want >=1 val record so the trainer can compute an
    eval loss for this bucket."""
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)])
    train, val = sample_offline_qa_records(p, val_ratio=0.1, seed=0)
    assert len(train) == 9
    assert len(val) == 1


def test_sample_val_ratio_zero_means_all_train(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(10)])
    train, val = sample_offline_qa_records(p, val_ratio=0.0, seed=0)
    assert len(train) == 10
    assert len(val) == 0


def test_sample_deterministic_same_seed_same_split(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)])
    t1, v1 = sample_offline_qa_records(p, val_ratio=0.2, seed=42)
    t2, v2 = sample_offline_qa_records(p, val_ratio=0.2, seed=42)
    assert [r["conversations"][0]["content"] for r in t1] == [
        r["conversations"][0]["content"] for r in t2
    ]
    assert [r["conversations"][0]["content"] for r in v1] == [
        r["conversations"][0]["content"] for r in v2
    ]


def test_sample_train_val_disjoint(tmp_path: Path) -> None:
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)])
    train, val = sample_offline_qa_records(p, val_ratio=0.2, seed=0)
    train_qs = {r["conversations"][0]["content"] for r in train}
    val_qs = {r["conversations"][0]["content"] for r in val}
    assert train_qs.isdisjoint(val_qs)


def test_sample_single_record_goes_to_train_no_val(tmp_path: Path) -> None:
    """Degenerate edge case: with only 1 record we can't split, so it
    goes to train and val is empty. The trainer just won't get an
    offline_qa eval signal but train still benefits."""
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": "Solo", "answer": "Only one."}])
    train, val = sample_offline_qa_records(p, val_ratio=0.1, seed=0)
    assert len(train) == 1
    assert len(val) == 0


def test_sample_real_corpus_size(tmp_path: Path) -> None:
    """Sanity check on the production corpus size: 42 records with
    val_ratio=0.1 → 4 val (floor(42*0.1)=4) + 38 train."""
    p = tmp_path / "qa.json"
    _write_qa(p, [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(42)])
    train, val = sample_offline_qa_records(p, val_ratio=0.1, seed=42)
    assert len(train) == 38
    assert len(val) == 4
