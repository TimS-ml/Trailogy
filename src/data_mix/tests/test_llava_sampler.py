"""Tests for the v2 llava_sampler (HuggingFaceH4/llava-instruct-mix-vsft).

Key behaviors vs the v1 cambrian_sampler this replaces:

- LLaVA-mix-vsft schema is different: ``messages[i].content`` is a LIST
  of ``{type: text|image, text?: str, index?: int}`` blocks, not a flat
  string. We flatten text blocks per turn into a single string and
  discard image blocks (we use the separate ``images`` field for the
  actual PIL data).
- Multi-turn conversations are PRESERVED (cambrian_sampler only kept
  the first user/assistant pair). iOS app runs 10-turn dialogues, so
  multi-turn exposure during SFT is desirable.
- No plant-like filter — user specifically requested non-plant filter
  off. LLaVA-mix has very few plant images, and the negative bucket
  refusal template handles any incidental leakage.
- Rows with ``images`` list length != 1 are dropped (single-image
  invariant matches the rest of the mix and avoids collator complexity).
"""
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data_mix.src.llava_sampler import (
    _first_turn_after_flatten,
    _flatten_content,
    sample_llava_records,
)
from data_mix.src.schema import validate_record


# ---------------------------------------------------------------------------
# Fake HF row builder (matches the LLaVA-mix-vsft observed schema)
# ---------------------------------------------------------------------------

def _content_text(text: str):
    return {"index": None, "type": "text", "text": text}


def _content_image(idx: int = 0):
    return {"index": idx, "type": "image", "text": None}


def _hf_row(
    rid: str,
    turns: list[tuple[str, list]],   # [(role, content_blocks), ...]
    n_images: int = 1,
):
    msgs = [{"role": role, "content": blocks} for role, blocks in turns]
    images = [
        Image.new("RGB", (200, 200), color=(i * 30 + 10, 50, 80))
        for i in range(n_images)
    ]
    return {"id": rid, "messages": msgs, "images": images}


# ---------------------------------------------------------------------------
# _flatten_content
# ---------------------------------------------------------------------------

def test_flatten_content_joins_text_blocks_and_drops_image():
    content = [
        _content_text("Who wrote this book?\n"),
        _content_image(0),
    ]
    assert _flatten_content(content) == "Who wrote this book?"


def test_flatten_content_image_can_appear_before_text():
    # Real LLaVA-mix rows put image first sometimes.
    content = [
        _content_image(0),
        _content_text("\nDescribe this image."),
    ]
    assert _flatten_content(content) == "Describe this image."


def test_flatten_content_multiple_text_blocks_joined():
    content = [
        _content_text("Part one.  "),
        _content_text("Part two."),
    ]
    assert _flatten_content(content) == "Part one.  Part two."


def test_flatten_content_returns_empty_when_no_text():
    assert _flatten_content([_content_image(0)]) == ""


def test_flatten_content_handles_string_content():
    # Defensive: some rows might already be flat string content (older
    # cached splits or future schema drift).
    assert _flatten_content("plain string") == "plain string"


# ---------------------------------------------------------------------------
# _first_turn_after_flatten (drops rows that don't have at least user+assistant)
# ---------------------------------------------------------------------------

def test_first_turn_after_flatten_returns_pair_when_present():
    msgs = [
        {"role": "user", "content": [_content_text("Q")]},
        {"role": "assistant", "content": [_content_text("A")]},
    ]
    got = _first_turn_after_flatten(msgs)
    assert got == ("Q", "A")


def test_first_turn_after_flatten_returns_none_if_only_one_turn():
    msgs = [{"role": "user", "content": [_content_text("only user")]}]
    assert _first_turn_after_flatten(msgs) is None


def test_first_turn_after_flatten_returns_none_on_empty_text():
    # Both text blocks empty after flatten -> drop.
    msgs = [
        {"role": "user", "content": [_content_image(0)]},  # no text
        {"role": "assistant", "content": [_content_text("A")]},
    ]
    assert _first_turn_after_flatten(msgs) is None


# ---------------------------------------------------------------------------
# sample_llava_records — main API
# ---------------------------------------------------------------------------

def _fake_stream(rows: list):
    """Yield rows verbatim (HF datasets stream is just an iterable)."""
    yield from rows


def test_sampler_splits_general_and_negative(tmp_path: Path):
    rows = [
        _hf_row("a", [
            ("user", [_content_text("What's this?"), _content_image(0)]),
            ("assistant", [_content_text("A book.")]),
        ]),
        _hf_row("b", [
            ("user", [_content_text("Read the sign.")]),
            ("assistant", [_content_text("STOP")]),
        ]),
        _hf_row("c", [
            ("user", [_content_text("Count items.")]),
            ("assistant", [_content_text("Three.")]),
        ]),
    ]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=2,
        n_negative=1,
        image_root=tmp_path / "img",
        seed=0,
    )
    assert len(res["general"]) == 2
    assert len(res["negative"]) == 1

    for rec in res["general"]:
        validate_record(rec)
        assert rec["source"] == "llava"
        assert Path(rec["image"]).exists()
        with Image.open(rec["image"]) as im:
            # Must be at trained vision shape.
            assert im.size == (672, 960)

    for p in res["negative"]:
        assert isinstance(p, Path)
        assert p.exists()
        with Image.open(p) as im:
            assert im.size == (672, 960)


def test_sampler_preserves_multi_turn(tmp_path: Path):
    """LLaVA-mix has rows with 4+ turns. v2 keeps them all (vs cambrian
    which truncated to 2)."""
    rows = [_hf_row("multi", [
        ("user",      [_content_text("Q1"), _content_image(0)]),
        ("assistant", [_content_text("A1")]),
        ("user",      [_content_text("Q2")]),
        ("assistant", [_content_text("A2")]),
        ("user",      [_content_text("Q3")]),
        ("assistant", [_content_text("A3")]),
    ])]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=1,
        n_negative=0,
        image_root=tmp_path / "img",
        seed=0,
    )
    assert len(res["general"]) == 1
    rec = res["general"][0]
    assert len(rec["conversations"]) == 6, (
        "multi-turn truncated — v2 should keep all turns"
    )
    # Roles alternate user/assistant/user/...
    roles = [t["role"] for t in rec["conversations"]]
    assert roles == ["user", "assistant"] * 3


def test_sampler_no_plant_filter(tmp_path: Path):
    """v2 explicitly removes the plant-like filter (per user spec).
    Plant-mentioning rows from LLaVA-mix MUST land in general."""
    rows = [_hf_row("plant_row", [
        ("user", [_content_text("What plant is in the garden?"), _content_image(0)]),
        ("assistant", [_content_text("A rose bush.")]),
    ])]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=1,
        n_negative=0,
        image_root=tmp_path / "img",
        seed=0,
    )
    assert len(res["general"]) == 1, "plant-mentioning row should not be filtered"


def test_sampler_skips_multi_image_rows(tmp_path: Path):
    """Records with images list length != 1 are skipped (single-image
    invariant)."""
    rows = [
        _hf_row("ok", [
            ("user", [_content_text("Q"), _content_image(0)]),
            ("assistant", [_content_text("A")]),
        ], n_images=1),
        _hf_row("two_imgs", [
            ("user", [_content_text("Q"), _content_image(0), _content_image(1)]),
            ("assistant", [_content_text("A")]),
        ], n_images=2),
        _hf_row("zero_imgs", [
            ("user", [_content_text("Q")]),
            ("assistant", [_content_text("A")]),
        ], n_images=0),
    ]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=10,
        n_negative=10,
        image_root=tmp_path / "img",
        seed=0,
    )
    # Only the single-image row survives.
    assert len(res["general"]) + len(res["negative"]) == 1


def test_sampler_short_stream_returns_what_it_has(tmp_path: Path):
    rows = [_hf_row(f"r{i}", [
        ("user", [_content_text(f"Q{i}"), _content_image(0)]),
        ("assistant", [_content_text(f"A{i}")]),
    ]) for i in range(3)]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=100,
        n_negative=100,
        image_root=tmp_path / "img",
        seed=0,
    )
    # 3 rows total. Filling general first (3), then negative (0).
    assert len(res["general"]) == 3
    assert len(res["negative"]) == 0


def test_sampler_general_filled_before_negative(tmp_path: Path):
    """When the stream is finite, general pool fills first to honor the
    bucket ratio."""
    rows = [_hf_row(f"r{i}", [
        ("user", [_content_text(f"Q{i}"), _content_image(0)]),
        ("assistant", [_content_text(f"A{i}")]),
    ]) for i in range(5)]
    res = sample_llava_records(
        stream=_fake_stream(rows),
        n_general=3,
        n_negative=10,  # we only have 5 rows, expect 5-3=2 in negative
        image_root=tmp_path / "img",
        seed=0,
    )
    assert len(res["general"]) == 3
    assert len(res["negative"]) == 2
