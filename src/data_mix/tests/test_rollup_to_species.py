"""Tests for the species-level rollup of enriched na_plantae records.

The fetcher pulls taxa at whatever rank iNaturalist publishes them
(species, subspecies, variety, form). For the on-device model we want
~800 binomial classes, not the raw ~956 mixed-rank taxa — see the
rollup design note for rationale.

This module verifies the merge logic without hitting the network.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_mix.src import rollup_to_species as rollup


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parent_row(sci="Acer negundo", slug="box_elder", common="box elder",
                wiki="Acer negundo, the box elder, is a species of maple.",
                gbif="", n_obs=50, n_photos=80) -> dict:
    return {
        "scientific_name": sci,
        "common_name": common,
        "slug": slug,
        "rank": "species",
        "n_observations": n_obs,
        "n_photos": n_photos,
        "wikipedia_summary": wiki,
        "wikipedia_title": sci,
        "wikipedia_url": f"https://en.wikipedia.org/wiki/{sci.replace(' ','_')}",
        "gbif_description": gbif,
        "gbif_description_source": "",
        "gbif_description_language": "",
        "gbif_distribution": "",
        "gbif_profile": "",
        "gbif_usage_key": "1",
        "gbif_url": "https://www.gbif.org/species/1",
        "common_names": common,
        "best_description": wiki or gbif,
        "best_description_source": "Wikipedia-en" if wiki else ("GBIF" if gbif else ""),
        "fetch_status": "ok" if (wiki or gbif) else "no_description_found",
        "rag_text": "",
    }


def _child_row(sci, slug="unknown", common="?", rank_="subspecies",
               wiki="", gbif="", n_obs=20, n_photos=30) -> dict:
    return {
        **_parent_row(sci, slug, common, wiki, gbif, n_obs, n_photos),
        "rank": rank_,
    }


# ---------------------------------------------------------------------------
# group_by_binomial
# ---------------------------------------------------------------------------

def test_binomial_alone_yields_single_group() -> None:
    rows = [_parent_row(sci="Acer rubrum", slug="red_maple")]
    groups = rollup.group_by_binomial(rows)
    assert list(groups.keys()) == ["Acer rubrum"]
    assert len(groups["Acer rubrum"]) == 1


def test_subspecies_grouped_under_parent_binomial() -> None:
    rows = [
        _parent_row(sci="Acer negundo", slug="box_elder"),
        _child_row("Acer negundo violaceum", slug="violet_boxelder"),
    ]
    groups = rollup.group_by_binomial(rows)
    assert list(groups.keys()) == ["Acer negundo"]
    assert {r["scientific_name"] for r in groups["Acer negundo"]} == {
        "Acer negundo", "Acer negundo violaceum",
    }


def test_orphan_trinomial_without_parent_still_grouped() -> None:
    """Even if the binomial parent is absent we shouldn't drop the
    trinomial — the rollup should promote it to fill the slot.
    """
    rows = [_child_row("Foo bar baz", slug="something", rank_="variety")]
    groups = rollup.group_by_binomial(rows)
    assert "Foo bar" in groups


# ---------------------------------------------------------------------------
# rollup_group: merge a list of rows into one binomial-level record
# ---------------------------------------------------------------------------

def test_rollup_keeps_parent_metadata_when_present() -> None:
    parent = _parent_row(sci="Acer negundo", slug="box_elder",
                         common="box elder",
                         wiki="Acer negundo, the box elder, is a species of maple.")
    child = _child_row("Acer negundo violaceum", slug="violet_boxelder",
                       common="violet boxelder")
    merged = rollup.rollup_group("Acer negundo", [parent, child])
    assert merged["scientific_name"] == "Acer negundo"
    assert merged["slug"] == "box_elder"
    assert merged["common_name"] == "box elder"
    assert merged["best_description_source"] == "Wikipedia-en"
    assert "box elder" in merged["best_description"]


def test_rollup_records_child_taxa_and_sums_counts() -> None:
    parent = _parent_row(sci="Acer negundo", slug="box_elder",
                         n_obs=50, n_photos=80)
    child_a = _child_row("Acer negundo violaceum", n_obs=20, n_photos=30)
    child_b = _child_row("Acer negundo californicum", n_obs=10, n_photos=15)
    merged = rollup.rollup_group("Acer negundo", [parent, child_a, child_b])
    assert merged["n_observations"] == 80
    assert merged["n_photos"] == 125
    assert merged["child_taxa"] == [
        "Acer negundo californicum", "Acer negundo violaceum",
    ]


def test_rollup_fills_parent_description_from_child_when_parent_empty() -> None:
    """A binomial that hit no_description_found should adopt any child's
    description before being marked empty."""
    parent = _parent_row(sci="Foo bar", wiki="", gbif="")
    child = _child_row("Foo bar baz", wiki="A useful description.")
    merged = rollup.rollup_group("Foo bar", [parent, child])
    assert merged["best_description"] == "A useful description."
    assert merged["best_description_source"] == "Wikipedia-en"
    assert merged["fetch_status"] == "ok"
    assert merged["filled_from_child"] == "Foo bar baz"


def test_rollup_promotes_child_when_no_parent_binomial() -> None:
    child = _child_row("Foo bar baz", slug="foo_var", common="foo var",
                       wiki="Some text.")
    merged = rollup.rollup_group("Foo bar", [child])
    assert merged["scientific_name"] == "Foo bar"
    assert merged["common_name"] == "foo var"
    assert merged["slug"] == "foo_var"
    assert merged["child_taxa"] == ["Foo bar baz"]
    assert merged["promoted_from_child"] is True


def test_rollup_replaces_unknown_slug_with_child_slug() -> None:
    parent = _parent_row(sci="Acer foo", slug="unknown", common="?")
    child = _child_row("Acer foo bar", slug="useful_slug",
                       common="useful common")
    merged = rollup.rollup_group("Acer foo", [parent, child])
    assert merged["slug"] == "useful_slug"
    assert merged["common_name"] == "useful common"


# ---------------------------------------------------------------------------
# end-to-end main(): reads enriched + docs JSONLs and writes the rolled output
# ---------------------------------------------------------------------------

def test_main_writes_rolled_jsonls(tmp_path: Path) -> None:
    enriched = tmp_path / "species_enriched.jsonl"
    docs = tmp_path / "species_rag_docs.jsonl"
    out_enriched = tmp_path / "species_enriched_rolled.jsonl"
    out_docs = tmp_path / "species_rag_docs_rolled.jsonl"

    _write_jsonl(enriched, [
        _parent_row(sci="Acer negundo", slug="box_elder",
                    wiki="Acer negundo is a maple."),
        _child_row("Acer negundo violaceum", slug="violet_boxelder"),
        _parent_row(sci="Acer rubrum", slug="red_maple",
                    wiki="Acer rubrum is a maple."),
    ])
    _write_jsonl(docs, [
        {"id": "na_plantae:box_elder:x", "scientific_name": "Acer negundo",
         "text": "old", "metadata": {}},
        {"id": "na_plantae:violet_boxelder:y",
         "scientific_name": "Acer negundo violaceum",
         "text": "old", "metadata": {}},
        {"id": "na_plantae:red_maple:z", "scientific_name": "Acer rubrum",
         "text": "old", "metadata": {}},
    ])

    rc = rollup.main([
        "--input-enriched", str(enriched),
        "--input-docs", str(docs),
        "--output-enriched", str(out_enriched),
        "--output-docs", str(out_docs),
    ])
    assert rc == 0

    enriched_rows = [
        json.loads(line) for line in out_enriched.read_text().splitlines()
    ]
    docs_rows = [
        json.loads(line) for line in out_docs.read_text().splitlines()
    ]
    assert len(enriched_rows) == 2
    assert len(docs_rows) == 2
    assert {r["scientific_name"] for r in enriched_rows} == {
        "Acer negundo", "Acer rubrum",
    }
    # IDs should be unique and stable per binomial
    assert len({d["id"] for d in docs_rows}) == 2
