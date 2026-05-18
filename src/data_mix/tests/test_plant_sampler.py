from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from data_mix.src.plant_sampler import (
    PROMPT_VARIANTS,
    extract_species_id,
    sample_plant_records,
    sample_plant_records_split,
)
from data_mix.src.schema import validate_record


def _row(sid: str, hsh: str, common: str = "Eastern Hemlock"):
    return {
        "image": f"/data/images_resized/train/{sid}/{hsh}.jpg",
        "conversations": [
            {"role": "user", "content": "Can you identify this species?"},
            {"role": "assistant",
             "content": f"This is {common}. A small evergreen tree."},
        ],
    }


def test_prompt_variants_are_three():
    assert len(PROMPT_VARIANTS) == 3
    assert any("species" in p.lower() for p in PROMPT_VARIANTS)


def test_extract_species_id_basic():
    assert extract_species_id(_row("0123", "abc")["image"]) == "0123"


def test_per_class_cap_enforced(tmp_jsonl):
    rows = []
    for sid in range(5):
        for k in range(40):
            rows.append(_row(f"{sid:04d}", f"h{k}"))
    p = tmp_jsonl(rows)

    out = sample_plant_records(
        jsonl_path=p, total=200, per_class_cap=10, seed=0
    )
    assert len(out) == 50  # 5 classes * cap 10
    counts = Counter(extract_species_id(r["image"]) for r in out)
    assert all(v == 10 for v in counts.values())


def test_total_clip_when_pool_large(tmp_jsonl):
    rows = [_row("0001", f"h{k}") for k in range(500)]
    p = tmp_jsonl(rows)
    out = sample_plant_records(p, total=50, per_class_cap=1000, seed=0)
    assert len(out) == 50


def test_records_validate(tmp_jsonl):
    rows = [_row(f"{i:04d}", "x") for i in range(20)]
    p = tmp_jsonl(rows)
    out = sample_plant_records(p, total=20, per_class_cap=50, seed=42)
    for rec in out:
        validate_record(rec)
        assert rec["source"] == "plant"


def test_prompt_distribution_roughly_uniform(tmp_jsonl):
    rows = [_row(f"{i:04d}", "x") for i in range(3000)]
    p = tmp_jsonl(rows)
    out = sample_plant_records(p, total=3000, per_class_cap=10_000, seed=0)
    cnt = Counter(r["conversations"][0]["content"] for r in out)
    assert set(cnt.keys()) == set(PROMPT_VARIANTS)
    # Each variant should be 33% +- 5%
    for v in cnt.values():
        assert 0.28 * 3000 <= v <= 0.38 * 3000


def test_third_variant_uses_full_desc(tmp_jsonl):
    rows = [_row(f"{i:04d}", "x", common="Fern") for i in range(2000)]
    p = tmp_jsonl(rows)
    out = sample_plant_records(p, total=2000, per_class_cap=5000, seed=0)
    detail_prompt = PROMPT_VARIANTS[2]
    detail = [r for r in out if r["conversations"][0]["content"] == detail_prompt]
    assert detail, "expected at least one detail-prompt sample"
    # Detail-prompt responses must include the full original assistant text.
    for r in detail:
        assert "A small evergreen tree" in r["conversations"][1]["content"]


# ---------------------------------------------------------------------------
# sample_plant_records_split — v2 dual-source API (train.jsonl + val.jsonl)
#
# The v1 sample_plant_records used a single jsonl + per_class_cap then mix.py
# random-sliced the pool into train/val. That re-introduced the very split-
# bias problem the upstream prepare-script fix was meant to solve. v2 reads
# train.jsonl and val.jsonl SEPARATELY — each pool has its own per_class_cap
# applied, no random slicing between them.
# ---------------------------------------------------------------------------

def test_split_reads_train_and_val_jsonl_separately(tmp_jsonl, tmp_path):
    """Mix train pool comes from train.jsonl, mix val pool from val.jsonl —
    no record from train.jsonl appears in val output and vice versa."""
    train_rows = [_row(f"{i:04d}", f"train_{k}") for i in range(10) for k in range(5)]
    val_rows = [_row(f"{i:04d}", f"val_{k}") for i in range(10) for k in range(2)]
    train_p = tmp_jsonl(train_rows, name="train.jsonl")
    val_p = tmp_jsonl(val_rows, name="val.jsonl")

    train_out, val_out = sample_plant_records_split(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=30, n_val=15,
        per_class_cap=10, seed=0,
    )
    assert len(train_out) == 30
    assert len(val_out) == 15

    train_images = {r["image"] for r in train_out}
    val_images = {r["image"] for r in val_out}
    assert train_images.isdisjoint(val_images), (
        "train and val outputs share images — pools leaked"
    )
    # Every train output came from train.jsonl
    for r in train_out:
        assert "train_" in r["image"], (
            f"train output {r['image']} not from train.jsonl"
        )
    for r in val_out:
        assert "val_" in r["image"], (
            f"val output {r['image']} not from val.jsonl"
        )


def test_split_per_class_cap_applies_to_each_pool(tmp_jsonl):
    """per_class_cap is applied to BOTH train and val pools independently."""
    train_rows = [_row(f"{i:04d}", f"t{k}") for i in range(5) for k in range(40)]
    val_rows = [_row(f"{i:04d}", f"v{k}") for i in range(5) for k in range(40)]
    train_p = tmp_jsonl(train_rows, name="train.jsonl")
    val_p = tmp_jsonl(val_rows, name="val.jsonl")

    # cap=10 with 5 classes => 50 capped pool max. Ask for 40+20 — both fit.
    train_out, val_out = sample_plant_records_split(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=40, n_val=20,
        per_class_cap=10, seed=0,
    )
    assert len(train_out) == 40
    train_counts = Counter(extract_species_id(r["image"]) for r in train_out)
    # Cap caps every class at 10 before slicing; n_train=40 then takes 40
    # rows. Class distribution is roughly even but not strictly uniform
    # after the second shuffle in _cap_pool_by_class.
    assert all(v <= 10 for v in train_counts.values()), train_counts
    assert sum(train_counts.values()) == 40
    # Val pool similarly capped at 5*10=50, then n_val=20 clips.
    assert len(val_out) == 20


def test_split_emits_source_plant_in_both_outputs(tmp_jsonl):
    train_rows = [_row(f"{i:04d}", f"t{k}") for i in range(3) for k in range(5)]
    val_rows = [_row(f"{i:04d}", f"v{k}") for i in range(3) for k in range(5)]
    train_p = tmp_jsonl(train_rows, name="train.jsonl")
    val_p = tmp_jsonl(val_rows, name="val.jsonl")

    train_out, val_out = sample_plant_records_split(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=10, n_val=5,
        per_class_cap=5, seed=0,
    )
    for r in train_out + val_out:
        validate_record(r)
        assert r["source"] == "plant"


def test_split_is_deterministic(tmp_jsonl):
    """Same inputs + same seed -> bit-identical output."""
    train_rows = [_row(f"{i:04d}", f"t{k}") for i in range(8) for k in range(5)]
    val_rows = [_row(f"{i:04d}", f"v{k}") for i in range(8) for k in range(3)]
    train_p = tmp_jsonl(train_rows, name="train.jsonl")
    val_p = tmp_jsonl(val_rows, name="val.jsonl")

    train_a, val_a = sample_plant_records_split(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=20, n_val=10, per_class_cap=5, seed=42,
    )
    train_b, val_b = sample_plant_records_split(
        train_jsonl=train_p, val_jsonl=val_p,
        n_train=20, n_val=10, per_class_cap=5, seed=42,
    )
    assert train_a == train_b
    assert val_a == val_b


def test_split_raises_when_pool_short(tmp_jsonl):
    """If either pool is too small to fill n_train or n_val, surface a
    clear error rather than silently returning fewer rows."""
    train_rows = [_row(f"{i:04d}", f"t{k}") for i in range(3) for k in range(5)]
    val_rows = [_row(f"{i:04d}", f"v{k}") for i in range(3) for k in range(2)]
    train_p = tmp_jsonl(train_rows, name="train.jsonl")
    val_p = tmp_jsonl(val_rows, name="val.jsonl")

    # Asks for n_val=100 but val pool has only 6 (3 classes * 2 imgs, even
    # with no cap).
    with pytest.raises(ValueError) as exc:
        sample_plant_records_split(
            train_jsonl=train_p, val_jsonl=val_p,
            n_train=10, n_val=100, per_class_cap=10, seed=0,
        )
    assert "val" in str(exc.value).lower()
