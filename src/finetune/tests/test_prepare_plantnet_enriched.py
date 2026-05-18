"""Tests for src.prepare_plantnet_enriched.

Cover the new behaviors layered on top of prepare_plantnet:
  * English vernacular extraction from a synthetic GBIF cache.
  * First-sentence + cap-at-N description trimming.
  * Filter pipeline (no metadata / no English / no description).
  * End-to-end smoke: synthetic PlantNet tree + CSV + cache → JSONL + report.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pytest

from src.prepare_plantnet_enriched import (
    ANSWER_TEMPLATES_RICH,
    QUESTION_TEMPLATES,
    build_enriched_conversation,
    build_species_index,
    build_taxonkey_to_english_index,
    extract_first_sentence,
    pick_english_vernacular,
    pick_first_common_name,
    SpeciesInfo,
)


# ---------------------------------------------------------------------------
# pick_english_vernacular
# ---------------------------------------------------------------------------


def test_pick_english_prefers_lang_eng() -> None:
    items = [
        {"vernacularName": "Souci étoilé", "language": "fra"},
        {"vernacularName": "Star Marigold", "language": "eng"},
        {"vernacularName": "Stern-Ringelblume", "language": "deu"},
    ]
    assert pick_english_vernacular(items) == "Star Marigold"


def test_pick_english_handles_en_suffix_with_blank_language() -> None:
    # Catalogue of Life pattern: blank language, "(EN)" suffix on the name.
    items = [
        {"vernacularName": "Tall Tutsan (EN)", "language": ""},
        {"vernacularName": "duftloses Johanniskraut", "language": "deu"},
    ]
    assert pick_english_vernacular(items) == "Tall Tutsan"


def test_pick_english_returns_none_when_no_english() -> None:
    items = [
        {"vernacularName": "Cirse de Montpellier", "language": "fra"},
        {"vernacularName": "Cardo di Montpellier", "language": "ita"},
        {"vernacularName": "Cardo", "language": "spa"},
    ]
    assert pick_english_vernacular(items) is None


def test_pick_english_titlecases_lowercase_names() -> None:
    # GBIF often returns "wild lettuce" all-lowercase from one source and
    # "Wild Lettuce" Title Case from another. We always Title-Case.
    items = [{"vernacularName": "wild lettuce", "language": "eng"}]
    assert pick_english_vernacular(items) == "Wild Lettuce"


def test_pick_english_preserves_mixed_case_words() -> None:
    # Don't mangle words like "McKinley" — only flip ALL-LOWER/ALL-UPPER.
    items = [{"vernacularName": "McKinley fir", "language": "eng"}]
    assert pick_english_vernacular(items) == "McKinley Fir"


def test_pick_english_skips_blank_entries() -> None:
    items = [
        {"vernacularName": "", "language": "eng"},
        {"vernacularName": "Real Name", "language": "eng"},
    ]
    assert pick_english_vernacular(items) == "Real Name"


# ---------------------------------------------------------------------------
# pick_first_common_name (V2 path)
# ---------------------------------------------------------------------------


def test_pick_first_common_name_basic() -> None:
    assert pick_first_common_name("Acrid lettuce; Bitter lettuce; Great Lettuce") == "Acrid Lettuce"


def test_pick_first_common_name_titlecases() -> None:
    assert pick_first_common_name("wild lettuce; bitter lettuce") == "Wild Lettuce"


def test_pick_first_common_name_single_entry() -> None:
    assert pick_first_common_name("Eastern Hemlock") == "Eastern Hemlock"


def test_pick_first_common_name_empty() -> None:
    assert pick_first_common_name("") is None
    assert pick_first_common_name("   ") is None
    assert pick_first_common_name("; ;  ; ") is None


def test_pick_first_common_name_none() -> None:
    assert pick_first_common_name(None) is None


def test_pick_first_common_name_skips_leading_semicolons() -> None:
    assert pick_first_common_name("; ; Coast Storksbill; rose geranium") == "Coast Storksbill"


# ---------------------------------------------------------------------------
# build_taxonkey_to_english_index
# ---------------------------------------------------------------------------


def _write_cache_file(cache_dir: Path, name: str, payload: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / name).write_text(json.dumps(payload), encoding="utf-8")


def test_index_skips_files_without_taxonkey(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    _write_cache_file(
        cache,
        "gbif_vernacularNames_aaaa.json",
        {"results": []},  # no results → skipped
    )
    _write_cache_file(
        cache,
        "gbif_vernacularNames_bbbb.json",
        {
            "results": [
                {"taxonKey": 12345, "vernacularName": "Hemlock", "language": "eng"}
            ]
        },
    )
    index = build_taxonkey_to_english_index(cache)
    assert index == {"12345": "Hemlock"}


def test_index_ignores_other_cache_file_types(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    # Should be ignored (different filename prefix).
    _write_cache_file(
        cache,
        "gbif_descriptions_dead.json",
        {"results": [{"taxonKey": 99, "vernacularName": "should not", "language": "eng"}]},
    )
    _write_cache_file(
        cache,
        "gbif_vernacularNames_beef.json",
        {"results": [{"taxonKey": 42, "vernacularName": "kept", "language": "eng"}]},
    )
    index = build_taxonkey_to_english_index(cache)
    assert index == {"42": "Kept"}


def test_index_returns_empty_when_cache_missing(tmp_path: Path) -> None:
    # No exception, just empty + a warning (which we don't assert).
    index = build_taxonkey_to_english_index(tmp_path / "nope")
    assert index == {}


def test_index_skips_unparseable_files(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "gbif_vernacularNames_zzz.json").write_text("not json{")
    _write_cache_file(
        cache,
        "gbif_vernacularNames_good.json",
        {"results": [{"taxonKey": 1, "vernacularName": "Survivor", "language": "eng"}]},
    )
    index = build_taxonkey_to_english_index(cache)
    assert index == {"1": "Survivor"}


# ---------------------------------------------------------------------------
# extract_first_sentence
# ---------------------------------------------------------------------------


def test_first_sentence_basic_period() -> None:
    text = "Tsuga canadensis is a coniferous tree. It is the state tree of Pennsylvania."
    assert extract_first_sentence(text, 400) == "Tsuga canadensis is a coniferous tree."


def test_first_sentence_collapses_internal_whitespace() -> None:
    # GBIF descriptions often contain literal "\r\r" paragraph breaks.
    text = "Annual plant.\r\r\tIt grows up to 200 cm. Lots more text..."
    out = extract_first_sentence(text, 400)
    assert out == "Annual plant."


def test_first_sentence_handles_no_period() -> None:
    text = "Just a short note without punctuation"
    assert extract_first_sentence(text, 400) == "Just a short note without punctuation"


def test_first_sentence_truncates_long_sentence_at_word_boundary() -> None:
    text = "This is " + "really " * 100 + "long."  # ~750 chars, no internal period.
    out = extract_first_sentence(text, 50)
    assert len(out) <= 50
    assert out.endswith("…")
    # Should not split a word: penultimate char before "…" should not be
    # mid-word — i.e. the truncated text ends on "really" boundary.
    body = out[:-1].rstrip()
    assert body.endswith("really")


def test_first_sentence_empty_input() -> None:
    assert extract_first_sentence("", 400) == ""
    assert extract_first_sentence("   \n\t  ", 400) == ""


def test_first_sentence_question_or_exclamation() -> None:
    text = "Have you seen this? It blooms in spring."
    assert extract_first_sentence(text, 400) == "Have you seen this?"


def test_first_sentence_doesnt_split_on_abbreviation_period() -> None:
    # "var." is followed by lowercase, so the regex (which requires
    # uppercase/digit after the period) doesn't split.
    text = "Lactuca virosa var. cruenta has broad lobes."
    out = extract_first_sentence(text, 400)
    assert out == "Lactuca virosa var. cruenta has broad lobes."


# ---------------------------------------------------------------------------
# build_species_index
# ---------------------------------------------------------------------------


def _write_enriched_csv(path: Path, rows: list[dict], *, v2: bool = False) -> None:
    """Write a minimal enriched CSV with all required columns.

    If ``v2=True``, include the ``common_names`` column (V2 format) and
    omit ``gbif_usage_key`` from the required set. Otherwise use the V1
    format with ``gbif_usage_key``.
    """
    if v2:
        cols = [
            "species_id",
            "species",
            "common_names",
            "wikipedia_summary",
            "best_description",
        ]
    else:
        cols = [
            "species_id",
            "species",
            "gbif_usage_key",
            "wikipedia_summary",
            "best_description",
        ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            row = {c: "" for c in cols}
            row.update(r)
            w.writerow(row)


def test_build_species_index_keeps_full_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "0",
                "species": "Tsuga canadensis",
                "gbif_usage_key": "111",
                "wikipedia_summary": "Tsuga canadensis is a coniferous tree native to eastern North America.",
                "best_description": "Lectotype: Clayton 547.",  # unused (Wikipedia wins)
            }
        ],
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={"111": "Eastern Hemlock"},
        description_max_chars=400,
        known_species_ids={"0"},
    )
    assert set(kept) == {"0"}
    assert kept["0"].common_name == "Eastern Hemlock"
    # Wikipedia preferred, first sentence kept.
    assert kept["0"].description == (
        "Tsuga canadensis is a coniferous tree native to eastern North America."
    )
    assert report.kept == ["0"]
    assert report.dropped_no_metadata == []
    assert report.dropped_no_english == []
    assert report.dropped_no_description == []


def test_build_species_index_drops_no_english(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "5",
                "species": "Calendula stellata",
                "gbif_usage_key": "555",
                "wikipedia_summary": "A Mediterranean annual.",
                "best_description": "",
            }
        ],
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={},  # no English available for this taxon
        description_max_chars=400,
    )
    assert kept == {}
    assert report.dropped_no_english == [("5", "Calendula stellata")]


def test_build_species_index_falls_back_to_gbif_description(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "1",
                "species": "Acer rubrum",
                "gbif_usage_key": "222",
                "wikipedia_summary": "",  # missing → fall back
                "best_description": "Acer rubrum is a common deciduous tree of eastern North America.",
            }
        ],
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={"222": "Red Maple"},
        description_max_chars=400,
    )
    assert kept["1"].description.startswith("Acer rubrum is a common deciduous tree")
    assert report.dropped_no_description == []


def test_build_species_index_drops_no_description(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "9",
                "species": "Genus species",
                "gbif_usage_key": "999",
                "wikipedia_summary": "",
                "best_description": "",
            }
        ],
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={"999": "Some Name"},
        description_max_chars=400,
    )
    assert kept == {}
    assert report.dropped_no_description == [("9", "Genus species")]


def test_build_species_index_records_known_but_missing_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "1",
                "species": "Foo bar",
                "gbif_usage_key": "111",
                "wikipedia_summary": "Foo bar is a thing.",
                "best_description": "",
            }
        ],
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={"111": "Foo"},
        description_max_chars=400,
        known_species_ids={"1", "2", "3"},  # 2 and 3 have no CSV row
    )
    assert set(kept) == {"1"}
    assert report.dropped_no_metadata == ["2", "3"]


def test_build_species_index_caps_long_description(tmp_path: Path) -> None:
    # 1000-char description — should be capped at 50 + ellipsis.
    long_desc = "Word " * 200  # 1000 chars
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "0",
                "species": "Test species",
                "gbif_usage_key": "1",
                "wikipedia_summary": long_desc,
                "best_description": "",
            }
        ],
    )
    kept, _ = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={"1": "Test"},
        description_max_chars=50,
    )
    assert len(kept["0"].description) <= 50


def test_build_species_index_rejects_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_species_index(
            enriched_csv=tmp_path / "does_not_exist.csv",
            taxonkey_to_english={},
            description_max_chars=400,
        )


def test_build_species_index_rejects_csv_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    csv_path.write_text("species_id,species\n0,Foo\n")  # missing required cols
    with pytest.raises(ValueError, match="missing required columns"):
        build_species_index(
            enriched_csv=csv_path,
            taxonkey_to_english={},
            description_max_chars=400,
        )


# ---------------------------------------------------------------------------
# build_species_index — V2 mode (use_csv_common_names=True)
# ---------------------------------------------------------------------------


def test_build_species_index_v2_keeps_full_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "0",
                "species": "Lactuca virosa",
                "common_names": "Acrid lettuce; Bitter lettuce; Great Lettuce",
                "wikipedia_summary": "Lactuca virosa is a plant in the lettuce genus.",
                "best_description": "Unused because Wikipedia wins.",
            }
        ],
        v2=True,
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={},  # unused in V2 mode
        description_max_chars=400,
        known_species_ids={"0"},
        use_csv_common_names=True,
    )
    assert set(kept) == {"0"}
    assert kept["0"].common_name == "Acrid Lettuce"
    assert "lettuce genus" in kept["0"].description
    assert report.kept == ["0"]
    assert report.dropped_no_english == []


def test_build_species_index_v2_drops_no_common_names(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "71",
                "species": "Hypericum empetrifolium",
                "common_names": "",  # no common names
                "wikipedia_summary": "A small shrub.",
                "best_description": "",
            }
        ],
        v2=True,
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={},
        description_max_chars=400,
        use_csv_common_names=True,
    )
    assert kept == {}
    assert report.dropped_no_english == [("71", "Hypericum empetrifolium")]


def test_build_species_index_v2_drops_no_description(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "5",
                "species": "Test plant",
                "common_names": "Test Common",
                "wikipedia_summary": "",
                "best_description": "",
            }
        ],
        v2=True,
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={},
        description_max_chars=400,
        use_csv_common_names=True,
    )
    assert kept == {}
    assert report.dropped_no_description == [("5", "Test plant")]


def test_build_species_index_v2_records_missing_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "0",
                "species": "Foo",
                "common_names": "Foo Plant",
                "wikipedia_summary": "Foo is a thing.",
                "best_description": "",
            }
        ],
        v2=True,
    )
    kept, report = build_species_index(
        enriched_csv=csv_path,
        taxonkey_to_english={},
        description_max_chars=400,
        known_species_ids={"0", "1", "2"},
        use_csv_common_names=True,
    )
    assert set(kept) == {"0"}
    assert sorted(report.dropped_no_metadata) == ["1", "2"]


def test_build_species_index_v2_rejects_csv_missing_common_names_col(tmp_path: Path) -> None:
    csv_path = tmp_path / "enriched.csv"
    # Write V1 format CSV but try to use it in V2 mode → should fail.
    _write_enriched_csv(csv_path, [{"species_id": "0", "species": "Foo"}], v2=False)
    with pytest.raises(ValueError, match="missing required columns"):
        build_species_index(
            enriched_csv=csv_path,
            taxonkey_to_english={},
            description_max_chars=400,
            use_csv_common_names=True,
        )


# ---------------------------------------------------------------------------
# build_enriched_conversation
# ---------------------------------------------------------------------------


def test_conversation_format_matches_legacy_schema(tmp_path: Path) -> None:
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")  # not a real JPEG; we only need the path
    info = SpeciesInfo(
        species_id="0",
        common_name="Eastern Hemlock",
        description="Tsuga canadensis is a coniferous tree.",
        latin="Tsuga canadensis",
    )
    rng = random.Random(0)
    rec = build_enriched_conversation(img, info, rng)
    # Schema parity with prepare_plantnet output.
    assert set(rec) == {"image", "conversations"}
    assert Path(rec["image"]).name == "img.jpg"
    assert len(rec["conversations"]) == 2
    assert rec["conversations"][0]["role"] == "user"
    assert rec["conversations"][0]["content"] in QUESTION_TEMPLATES
    assert rec["conversations"][1]["role"] == "assistant"
    answer = rec["conversations"][1]["content"]
    # The English common name must appear; Latin should NOT (per design).
    assert "Eastern Hemlock" in answer
    assert "Tsuga canadensis" in answer  # this comes from the description, not the intro
    # Description is included after the species name.
    assert "coniferous tree" in answer


def test_conversation_uses_ANSWER_TEMPLATES_RICH(tmp_path: Path) -> None:
    """Every emitted answer must be derivable from one of our templates."""
    img = tmp_path / "img.jpg"
    img.write_bytes(b"\xff\xd8")
    info = SpeciesInfo(
        species_id="0",
        common_name="Sugar Maple",
        description="Acer saccharum is best known for maple syrup.",
        latin="Acer saccharum",
    )
    # Iterate enough times to hit all templates.
    rng = random.Random(42)
    answers = {
        build_enriched_conversation(img, info, rng)["conversations"][1]["content"]
        for _ in range(50)
    }
    # Each emitted answer must be one of the templates filled with our values.
    expected = {
        t.format(common=info.common_name, description=info.description)
        for t in ANSWER_TEMPLATES_RICH
    }
    assert answers <= expected


# ---------------------------------------------------------------------------
# End-to-end smoke (no images decoded, no model loaded)
# ---------------------------------------------------------------------------


def test_smoke_full_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic train tree + CSV + cache → main() should produce JSONL files."""
    import sys

    plantnet_root = tmp_path / "plantnet"
    train = plantnet_root / "train"
    for sid in ("0", "1", "2", "3"):
        d = train / sid
        d.mkdir(parents=True)
        # Minimal JPEG magic so PIL won't crash inside resize_image_to_disk.
        # We disable resize below so this only needs to be a recognizable file.
        (d / f"{sid}_a.jpg").write_bytes(b"\xff\xd8\xff\xe0placeholder")
        (d / f"{sid}_b.jpg").write_bytes(b"\xff\xd8\xff\xe0placeholder")

    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            # 0: full English + Wikipedia → keep
            {
                "species_id": "0", "species": "Tsuga canadensis", "gbif_usage_key": "111",
                "wikipedia_summary": "Tsuga canadensis is a coniferous tree native to eastern North America.",
                "best_description": "",
            },
            # 1: no English → drop
            {
                "species_id": "1", "species": "Calendula stellata", "gbif_usage_key": "222",
                "wikipedia_summary": "A Mediterranean annual.", "best_description": "",
            },
            # 2: no description anywhere → drop
            {
                "species_id": "2", "species": "Test plant", "gbif_usage_key": "333",
                "wikipedia_summary": "", "best_description": "",
            },
            # species_id "3" missing from CSV entirely → dropped_no_metadata
        ],
    )

    cache = tmp_path / "cache"
    _write_cache_file(
        cache, "gbif_vernacularNames_111.json",
        {"results": [{"taxonKey": 111, "vernacularName": "Eastern Hemlock", "language": "eng"}]},
    )
    _write_cache_file(
        cache, "gbif_vernacularNames_333.json",
        {"results": [{"taxonKey": 333, "vernacularName": "Test Common", "language": "eng"}]},
    )

    out = tmp_path / "out"
    argv = [
        "prep",
        "--plantnet_root", str(plantnet_root),
        "--enriched_csv", str(csv_path),
        "--species_cache", str(cache),
        "--data_version", "v1",
        "--output_dir", str(out),
        "--max_samples", "10",
        "--val_ratio", "0.5",
        "--resize_to", "none",  # skip PIL.resize on fake-image bytes
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from src.prepare_plantnet_enriched import main
    main()

    # JSONL files exist and only contain species 0.
    train_jsonl = out / "train.jsonl"
    val_jsonl = out / "val.jsonl"
    assert train_jsonl.exists()
    assert val_jsonl.exists()

    seen_species: set[str] = set()
    seen_common_names: set[str] = set()
    for path in (train_jsonl, val_jsonl):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            # Image path points back into the synthetic train tree.
            img_parent = Path(rec["image"]).parent.name
            seen_species.add(img_parent)
            answer = rec["conversations"][1]["content"]
            # Expect English-only answer (no Latin prefix in the intro).
            assert "Eastern Hemlock" in answer
            seen_common_names.add("Eastern Hemlock")

    # Only species "0" survived.
    assert seen_species == {"0"}
    assert seen_common_names == {"Eastern Hemlock"}

    # Filter report records the drops.
    report = json.loads((out / "filter_report.json").read_text())
    assert report["kept_count"] == 1
    assert "1" in [sid for sid, _ in report["dropped_no_english"]]
    assert "2" in [sid for sid, _ in report["dropped_no_description"]]
    assert "3" in report["dropped_no_metadata"]


def test_smoke_full_pipeline_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic V2 tree + CSV (with common_names column) → main() should produce JSONL."""
    import sys

    plantnet_root = tmp_path / "plantnet_v2"
    # V2 uses zero-padded class indices as folder names.
    # Use --split test since V2 train/ may be incomplete.
    test_dir = plantnet_root / "test"
    for sid in ("0000", "0001", "0002", "0003"):
        d = test_dir / sid
        d.mkdir(parents=True)
        (d / f"{sid}_a.jpg").write_bytes(b"\xff\xd8\xff\xe0placeholder")
        (d / f"{sid}_b.jpg").write_bytes(b"\xff\xd8\xff\xe0placeholder")

    csv_path = tmp_path / "species_metadata_enriched.csv"
    # V2 CSV uses UNPADDED species_id ("0", "1", ...) even though the
    # image folder names are zero-padded ("0000", "0001", ...). The script
    # normalizes folder names by stripping leading zeros to match.
    _write_enriched_csv(
        csv_path,
        [
            # 0 (folder "0000"): has common name + Wikipedia → keep
            {
                "species_id": "0", "species": "Lactuca virosa",
                "common_names": "Acrid lettuce; Bitter lettuce; Great Lettuce",
                "wikipedia_summary": "Lactuca virosa is a plant in the lettuce genus.",
                "best_description": "",
            },
            # 1 (folder "0001"): no common names → drop
            {
                "species_id": "1", "species": "Calendula stellata",
                "common_names": "",
                "wikipedia_summary": "A Mediterranean annual.", "best_description": "",
            },
            # 2 (folder "0002"): has common name but no description → drop
            {
                "species_id": "2", "species": "Test plant",
                "common_names": "Test Common",
                "wikipedia_summary": "", "best_description": "",
            },
            # species_id "3" (folder "0003") missing from CSV entirely → dropped_no_metadata
        ],
        v2=True,
    )

    out = tmp_path / "out"
    argv = [
        "prep",
        "--plantnet_root", str(plantnet_root),
        "--enriched_csv", str(csv_path),
        "--data_version", "v2",
        "--split", "test",
        "--output_dir", str(out),
        "--max_samples", "10",
        "--val_ratio", "0.5",
        "--resize_to", "none",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from src.prepare_plantnet_enriched import main
    main()

    # JSONL files exist and only contain species 0000.
    train_jsonl = out / "train.jsonl"
    val_jsonl = out / "val.jsonl"
    assert train_jsonl.exists()
    assert val_jsonl.exists()

    seen_species: set[str] = set()
    for path in (train_jsonl, val_jsonl):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            img_parent = Path(rec["image"]).parent.name
            seen_species.add(img_parent)
            answer = rec["conversations"][1]["content"]
            assert "Acrid Lettuce" in answer
            assert "lettuce genus" in answer

    # Folder "0000" is normalized to "0" internally, but the image path
    # still contains the original folder name "0000".
    assert seen_species == {"0000"}

    # Filter report uses the CSV's unpadded species_id values.
    report = json.loads((out / "filter_report.json").read_text())
    assert report["kept_count"] == 1
    assert "1" in [sid for sid, _ in report["dropped_no_english"]]
    assert "2" in [sid for sid, _ in report["dropped_no_description"]]
    assert "3" in report["dropped_no_metadata"]


def test_smoke_v2_with_train_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """V2 with --split train (default) also works when train/ exists."""
    import sys

    plantnet_root = tmp_path / "plantnet_v2"
    train_dir = plantnet_root / "train"
    for sid in ("0000",):
        d = train_dir / sid
        d.mkdir(parents=True)
        (d / f"{sid}_a.jpg").write_bytes(b"\xff\xd8\xff\xe0placeholder")

    csv_path = tmp_path / "enriched.csv"
    _write_enriched_csv(
        csv_path,
        [
            {
                "species_id": "0", "species": "Lactuca virosa",
                "common_names": "Wild Lettuce",
                "wikipedia_summary": "Lactuca virosa is a bitter plant.",
                "best_description": "",
            },
        ],
        v2=True,
    )

    out = tmp_path / "out"
    argv = [
        "prep",
        "--plantnet_root", str(plantnet_root),
        "--enriched_csv", str(csv_path),
        "--data_version", "v2",
        "--split", "train",
        "--output_dir", str(out),
        "--max_samples", "10",
        "--val_ratio", "0.5",
        "--resize_to", "none",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from src.prepare_plantnet_enriched import main
    main()

    train_jsonl = out / "train.jsonl"
    val_jsonl = out / "val.jsonl"
    assert train_jsonl.exists() or val_jsonl.exists()

    all_records = []
    for path in (train_jsonl, val_jsonl):
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    all_records.append(json.loads(line))

    assert len(all_records) >= 1
    assert "Wild Lettuce" in all_records[0]["conversations"][1]["content"]
