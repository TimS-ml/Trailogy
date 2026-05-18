from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data_mix.src.cambrian_sampler import (
    PLANT_NEGATIVE_FILTER_TOKENS,
    _persist_image,
    is_plant_like,
    sample_cambrian_records,
)
from data_mix.src.schema import validate_record


def _make_pil(size=(640, 480), color=(10, 20, 30)):
    return Image.new("RGB", size, color=color)


def _row(
    rid: str,
    user_text: str,
    asst_text: str,
    img=None,
    category: str = "general",
):
    return {
        "id": rid,
        "category": category,
        "image": img if img is not None else _make_pil(),
        "conversations": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ],
    }


def test_filter_tokens_lowercase():
    assert all(t == t.lower() for t in PLANT_NEGATIVE_FILTER_TOKENS)


@pytest.mark.parametrize(
    "text, expected",
    [
        # Positive — actual plant references at word boundaries.
        ("a beautiful red Plant in the garden", True),
        ("look at this Flower", True),
        ("botanical illustration", True),
        ("the trees are tall", True),
        ("yellow flowers in spring", True),
        ("leaves on the ground", True),
        # Negative — not plant references; v1 substring matching used to
        # false-positive on these.
        ("a cat on a couch", False),
        ("vehicle parts", False),
        ("a plantation worker", False),     # plant + ation
        ("plantain bread recipe", False),   # plant + ain
        ("kidney transplant", False),       # trans + plant
        ("treetop walkway", False),         # tree + top
        ("streetlight at dusk", False),     # contains "tree" substring only
        ("a leaflet from the doctor", False),  # leaf + let
        ("powerplant emissions", False),    # power + plant
    ],
)
def test_is_plant_like(text, expected):
    assert is_plant_like(text) is expected


def _fake_stream():
    yield _row("r1", "What is this?", "A car.", category="general")
    yield _row("r2", "Tell me about this plant.", "A rose.", category="general")
    yield _row("r3", "Read the sign.", "STOP", category="ocr")
    yield _row("r4", "Identify this flower.", "Daisy", category="general")
    yield _row("r5", "Count the objects.", "3", category="counting")


def test_sampler_split_general_vs_negative(tmp_path: Path):
    out_dir = tmp_path / "img"
    res = sample_cambrian_records(
        stream=_fake_stream(),
        n_general=2,
        n_negative=1,
        image_root=out_dir,
        seed=0,
    )
    assert len(res["general"]) == 2
    assert len(res["negative"]) == 1

    for rec in res["general"]:
        validate_record(rec)
        assert rec["source"] == "cambrian"
        assert Path(rec["image"]).exists()
        with Image.open(rec["image"]) as im:
            assert im.size == (672, 960)
        # General must NOT be plant-like (best-effort filter)
        joined = " ".join(t["content"] for t in rec["conversations"])
        assert not is_plant_like(joined)

    # ``negative`` is a List[Path] of resized image paths only — the
    # negative_builder constructs records from these in mix.py.
    for p in res["negative"]:
        assert isinstance(p, Path)
        assert p.exists()
        with Image.open(p) as im:
            assert im.size == (672, 960)


def test_sampler_short_stream_returns_what_it_has(tmp_path: Path):
    res = sample_cambrian_records(
        stream=_fake_stream(),
        n_general=100,
        n_negative=100,
        image_root=tmp_path / "img",
        seed=0,
    )
    # Stream has 5 rows; some are plant-like (r2 mentions "plant",
    # r4 mentions "flower"), so general gets at most 3 non-plant rows.
    assert 1 <= len(res["general"]) <= 4
    # Negative pool reuses the non-plant rows too.
    assert len(res["general"]) + len(res["negative"]) <= 5


def test_persist_image_is_idempotent(tmp_path: Path):
    """Re-calling _persist_image with the same dest must NOT rewrite the
    file (skip-if-exists fast path) — guards against wasted disk writes
    on partial-build re-runs.
    """
    dest = tmp_path / "cambrian" / "abc.jpg"
    pil = _make_pil(size=(640, 480), color=(50, 60, 70))

    _persist_image(pil, dest)
    assert dest.exists()
    mtime_first = dest.stat().st_mtime_ns
    size_first = dest.stat().st_size

    # Confirm the resized output is at the trained shape.
    with Image.open(dest) as im:
        assert im.size == (672, 960)

    # Second call with a DIFFERENT PIL must not touch dest.
    new_pil = _make_pil(size=(100, 100), color=(255, 0, 0))
    _persist_image(new_pil, dest)
    assert dest.stat().st_mtime_ns == mtime_first
    assert dest.stat().st_size == size_first

    # And the tmp `.raw.jpg` sidecar must not have been left behind.
    assert not (tmp_path / "cambrian" / "abc.raw.jpg").exists()
