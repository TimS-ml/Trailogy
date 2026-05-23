"""Tests for the ``--rollup-to-species`` mode of prepare_na_plantae.

This mode treats every fetched ``<split>/<slug>/`` image folder as
belonging to its binomial parent so the prepared training corpus has
~800 binomial classes instead of the 956 mixed-rank taxa iNaturalist
returns. See ``rollup_to_species.py`` for the rationale.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_mix.src import prepare_na_plantae as prep


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# load_rolled_descriptions
# ---------------------------------------------------------------------------

def test_load_rolled_descriptions_uses_binomial_slug_and_answer(
    tmp_path: Path,
) -> None:
    rolled = tmp_path / "rolled.jsonl"
    _write_jsonl(rolled, [
        {
            "scientific_name": "Acer negundo",
            "slug": "box_elder",
            "common_name": "box elder",
            "child_taxa": ["Acer negundo violaceum"],
            "best_description": "...",
            "fetch_status": "ok",
        },
    ])
    descs = prep.load_rolled_descriptions(rolled)
    assert set(descs) == {"box_elder"}
    rec = descs["box_elder"]
    assert rec["species"] == "Acer negundo"
    assert rec["common_name"] == "box elder"
    assert "box elder" in rec["answer"].lower()
    assert "Acer negundo" in rec["answer"]


def test_load_rolled_descriptions_falls_back_to_slug_when_common_empty(
    tmp_path: Path,
) -> None:
    rolled = tmp_path / "rolled.jsonl"
    _write_jsonl(rolled, [
        {
            "scientific_name": "Foo bar",
            "slug": "foo_bar",
            "common_name": "",
            "child_taxa": [],
        },
    ])
    descs = prep.load_rolled_descriptions(rolled)
    assert descs["foo_bar"]["common_name"] == "foo bar"


# ---------------------------------------------------------------------------
# build_slug_rewrite_map
# ---------------------------------------------------------------------------

def test_slug_rewrite_map_rewrites_child_slugs_to_parent(
    tmp_path: Path,
) -> None:
    obs = tmp_path / "observations.jsonl"
    _write_jsonl(obs, [
        {"scientific_name": "Acer negundo", "slug": "box_elder"},
        {"scientific_name": "Acer negundo violaceum", "slug": "violet_boxelder"},
        {"scientific_name": "Acer rubrum", "slug": "red_maple"},
    ])
    rolled_rows = [
        {
            "scientific_name": "Acer negundo",
            "slug": "box_elder",
            "child_taxa": ["Acer negundo violaceum"],
        },
        {
            "scientific_name": "Acer rubrum",
            "slug": "red_maple",
            "child_taxa": [],
        },
    ]
    rewrite = prep.build_slug_rewrite_map(rolled_rows, obs)
    assert rewrite == {"violet_boxelder": "box_elder"}


def test_slug_rewrite_map_skips_unknown_child_slugs(tmp_path: Path) -> None:
    """A child sci_name absent from observations.jsonl should not
    appear in the map — defensive against partial data."""
    obs = tmp_path / "observations.jsonl"
    _write_jsonl(obs, [
        {"scientific_name": "Acer negundo", "slug": "box_elder"},
    ])
    rolled_rows = [
        {
            "scientific_name": "Acer negundo",
            "slug": "box_elder",
            "child_taxa": ["Acer negundo violaceum"],
        },
    ]
    rewrite = prep.build_slug_rewrite_map(rolled_rows, obs)
    assert rewrite == {}


def test_slug_rewrite_map_skips_self_mapping(tmp_path: Path) -> None:
    """If the child's slug equals its parent's (iNat sometimes shares
    common-name slugs across parent + indicating-subspecies), don't
    emit a no-op self-mapping."""
    obs = tmp_path / "observations.jsonl"
    _write_jsonl(obs, [
        {"scientific_name": "Eriophyllum confertiflorum",
         "slug": "golden_yarrow"},
        {"scientific_name": "Eriophyllum confertiflorum confertiflorum",
         "slug": "golden_yarrow"},
    ])
    rolled_rows = [
        {
            "scientific_name": "Eriophyllum confertiflorum",
            "slug": "golden_yarrow",
            "child_taxa": ["Eriophyllum confertiflorum confertiflorum"],
        },
    ]
    rewrite = prep.build_slug_rewrite_map(rolled_rows, obs)
    assert rewrite == {}


# ---------------------------------------------------------------------------
# discover_species_images_split with slug_rewrite
# ---------------------------------------------------------------------------

def _make_image(path: Path) -> None:
    """Write a tiny valid JPEG so prepare's resize step doesn't choke."""
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(path, "JPEG")


def test_split_discovery_merges_child_slug_into_parent(
    tmp_path: Path,
) -> None:
    src = tmp_path / "source"
    _make_image(src / "train" / "box_elder" / "1_0.jpg")
    _make_image(src / "train" / "violet_boxelder" / "2_0.jpg")
    _make_image(src / "val" / "box_elder" / "3_0.jpg")

    out = prep.discover_species_images_split(
        source_root=src,
        slugs=["box_elder"],
        slug_rewrite={"violet_boxelder": "box_elder"},
    )
    train_files = [p.name for p in out["box_elder"].get("train", [])]
    assert sorted(train_files) == ["1_0.jpg", "2_0.jpg"]


# ---------------------------------------------------------------------------
# Integration test for main(--rollup-to-species ...)
# ---------------------------------------------------------------------------

def test_main_rollup_writes_binomial_jsonls(tmp_path: Path) -> None:
    src = tmp_path / "source"
    out = tmp_path / "out"
    _make_image(src / "train" / "box_elder" / "obs1.jpg")
    _make_image(src / "train" / "violet_boxelder" / "obs2.jpg")
    _make_image(src / "val" / "box_elder" / "obs3.jpg")
    _make_image(src / "train" / "red_maple" / "obs4.jpg")

    _write_jsonl(src / "observations.jsonl", [
        {"scientific_name": "Acer negundo", "slug": "box_elder",
         "observation_id": 1},
        {"scientific_name": "Acer negundo violaceum",
         "slug": "violet_boxelder", "observation_id": 2},
        {"scientific_name": "Acer rubrum", "slug": "red_maple",
         "observation_id": 4},
    ])

    rolled = src / "species_enriched_rolled.jsonl"
    _write_jsonl(rolled, [
        {
            "scientific_name": "Acer negundo",
            "slug": "box_elder",
            "common_name": "box elder",
            "child_taxa": ["Acer negundo violaceum"],
            "best_description": "...",
            "fetch_status": "ok",
        },
        {
            "scientific_name": "Acer rubrum",
            "slug": "red_maple",
            "common_name": "red maple",
            "child_taxa": [],
            "best_description": "...",
            "fetch_status": "ok",
        },
    ])

    rc = prep.main([
        "--source_root", str(src),
        "--output_dir", str(out),
        "--rollup-to-species", str(rolled),
        "--resize_to", "none",
        "--no-synthesize_missing",
        "--source_layout", "split",
        # Disable species caps for this rollup-focused test — the tiny
        # synthetic fixtures here have only 1-2 imgs per species.
        "--min-imgs-per-species", "0",
        "--max-imgs-per-species", "0",
    ])
    assert rc == 0

    train_rows = [
        json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()
    ]
    val_rows = [
        json.loads(line) for line in (out / "val.jsonl").read_text().splitlines()
    ]
    train_slugs = sorted({r["slug"] for r in train_rows})
    assert train_slugs == ["box_elder", "red_maple"]

    # Both box_elder and violet_boxelder images should land under the
    # binomial slug. Output image paths must point at the binomial
    # output folder, not the trinomial source folder.
    box_train = [r for r in train_rows if r["slug"] == "box_elder"]
    assert len(box_train) == 2
    for r in box_train:
        assert "/box_elder/" in r["image"]
        assert "/violet_boxelder/" not in r["image"]
        assert r["species"] == "Acer negundo"
        assert "box elder" in r["conversations"][1]["content"].lower()

    val_slugs = sorted({r["slug"] for r in val_rows})
    assert val_slugs == ["box_elder"]
