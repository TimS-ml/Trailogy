"""Tests for ``build_enriched_captions.py``.

Focus areas:
  * ``build_enriched_answer`` — eval-anchor preservation, field
    inclusion / exclusion, ``None``-enrichment fallback.
  * Helpers (``_dedupe_common_names``, ``_truncate_distribution``) —
    pure functions with clear edge cases.
  * ``_rebuild_row`` — image / slug / species / family / user-side
    preserved, only assistant turn modified, hard length cap kicks in
    when the rebuilt content exceeds ``MAX_CONTENT_CHARS``.
  * End-to-end CLI smoke — full main() over 2 tiny JSONLs, asserts
    the report file's counts.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from data_mix.src import build_enriched_captions as bec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestDedupeCommonNames:
    def test_returns_empty_for_none_or_blank(self) -> None:
        assert bec._dedupe_common_names(None, "rose") == []
        assert bec._dedupe_common_names("", "rose") == []
        assert bec._dedupe_common_names("   ", "rose") == []

    def test_drops_exact_case_variant_of_primary(self) -> None:
        # The case-only contract: "Common Pitcher Plant" is dropped
        # when primary is "common pitcher plant". Hyphen / spacing
        # variants ("red-osier" vs "red osier") survive — see the
        # docstring on _dedupe_common_names for the rationale.
        names = (
            "Common Pitcher Plant; common pitcher plant; "
            "northern pitcher plant"
        )
        out = bec._dedupe_common_names(names, "common pitcher plant")
        assert "Common Pitcher Plant" not in out
        assert "common pitcher plant" not in out
        assert "northern pitcher plant" in out

    def test_keeps_hyphen_variants(self) -> None:
        # Documents the current contract — hyphen / no-hyphen variants
        # are NOT collapsed. Lock-in test: if a future implementation
        # adds hyphen normalisation it should also touch this test.
        names = "red-osier dogwood; red osier dogwood; American dogwood"
        out = bec._dedupe_common_names(names, "red osier dogwood")
        assert "red-osier dogwood" in out
        assert "American dogwood" in out

    def test_preserves_order_and_dedupes_within_list(self) -> None:
        out = bec._dedupe_common_names(
            "balsam; Balsam; Canada Balsam; balsam",
            primary="balsam fir",
        )
        # The first "balsam" wins; later case-variant repeats are dropped
        # but the order of distinct names is preserved.
        assert out == ["balsam", "Canada Balsam"]


class TestTruncateDistribution:
    def test_returns_none_for_empty(self) -> None:
        assert bec._truncate_distribution(None, 5) is None
        assert bec._truncate_distribution("", 5) is None
        assert bec._truncate_distribution(";  ;", 5) is None

    def test_no_truncation_when_under_cap(self) -> None:
        out = bec._truncate_distribution("Alaska; Maine; Vermont", 5)
        assert out == "Alaska; Maine; Vermont"

    def test_appends_remainder_marker_when_over_cap(self) -> None:
        regions = "; ".join(f"R{i}" for i in range(20))
        out = bec._truncate_distribution(regions, 5)
        # 5 kept regions + 1 "and N more regions" tail.
        parts = out.split("; ")
        assert len(parts) == 6
        assert parts[:5] == ["R0", "R1", "R2", "R3", "R4"]
        assert parts[5] == "and 15 more regions"


# ---------------------------------------------------------------------------
# build_enriched_answer
# ---------------------------------------------------------------------------

class TestBuildEnrichedAnswer:
    def test_eval_anchor_is_first_sentence(self) -> None:
        # The eval scorer's species extractor depends on this exact
        # leading phrase. Regression here silently breaks every plant
        # eval, so it has its own dedicated test.
        out = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea",
            {
                "common_name": "balsam fir",
                "scientific_name": "Abies balsamea",
                "accepted_scientific_name": "Abies balsamea",
                "wikipedia_summary": "A North American fir.",
                "gbif_distribution": "Maine; Vermont",
            },
        )
        assert out.startswith("Looks like balsam fir to me. ")

    def test_falls_back_to_compact_when_enriched_is_none(self) -> None:
        out = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea", enriched=None,
        )
        # Compact fallback drops the legacy "...is a plant species
        # found in North America." tail per the v2-enrich design.
        assert out == (
            "Looks like balsam fir to me. "
            "Abies balsamea, commonly called balsam fir."
        )
        assert "plant species found in North America" not in out

    def test_compact_fallback_skips_scientific_clause_when_unknown(
        self,
    ) -> None:
        out = bec.build_enriched_answer(
            "mystery plant", "(unknown)", enriched=None,
        )
        assert out == "Looks like mystery plant to me."

    def test_emits_accepted_name_only_when_different(self) -> None:
        # accepted == scientific → simple form
        same = bec.build_enriched_answer(
            "balsam fir", "Abies balsamea",
            {"scientific_name": "Abies balsamea",
             "accepted_scientific_name": "Abies balsamea"},
        )
        assert "Scientific name: Abies balsamea." in same
        assert "accepted:" not in same

        # accepted != scientific → flag both
        diff = bec.build_enriched_answer(
            "Bailey acacia", "Acacia baileyana",
            {"scientific_name": "Acacia baileyana",
             "accepted_scientific_name": "Racosperma baileyanum"},
        )
        assert (
            "Scientific name: Acacia baileyana "
            "(accepted: Racosperma baileyanum)."
        ) in diff

    def test_excludes_gbif_description_and_profile(self) -> None:
        # The two noisy fields the v2-enrich design explicitly drops:
        # gbif_description (Latin typification etc.) and gbif_profile
        # (multilingual JSON / repeated habitat tokens). If a future
        # refactor adds them back, this test catches it.
        out = bec.build_enriched_answer(
            "common dandelion", "Taraxacum officinale",
            {
                "scientific_name": "Taraxacum officinale",
                "accepted_scientific_name": "Taraxacum officinale",
                "wikipedia_summary": "A common dandelion.",
                "gbif_description":
                    "Note: Farwell noted a nomen ambiguum...",
                "gbif_profile":
                    'habitat: Terrestrial; habitat: terrestrial; '
                    'lifeForm: {"lifeForm":["Árvore"]}',
                "gbif_distribution": "North America (PRESENT)",
            },
        )
        assert "nomen ambiguum" not in out
        assert "Terrestrial" not in out
        assert "Árvore" not in out
        # Sanity: the kept-fields ARE in there.
        assert "Wikipedia" not in out  # field name not echoed
        assert "A common dandelion." in out
        assert "North America (PRESENT)" in out


# ---------------------------------------------------------------------------
# build_lite_nameboost_answer
# ---------------------------------------------------------------------------

class TestBuildLiteNameboostAnswer:
    def test_eval_anchor_is_first_sentence(self) -> None:
        out = bec.build_lite_nameboost_answer(
            "balsam fir", "Abies balsamea",
            {
                "common_name": "balsam fir",
                "scientific_name": "Abies balsamea",
                "wikipedia_summary":
                    "Abies balsamea, the balsam fir, is a North American fir.",
                "gbif_distribution": "Maine; Vermont; Quebec",
            },
        )
        assert out.startswith("Looks like balsam fir to me. ")

    def test_repeats_common_name_at_least_three_times(self) -> None:
        out = bec.build_lite_nameboost_answer(
            "purple foxglove", "Digitalis purpurea",
            {
                "scientific_name": "Digitalis purpurea",
                "wikipedia_summary":
                    "Digitalis purpurea is a toxic plant. It grows in temperate Europe.",
                "gbif_distribution": "Europe; North America; Asia",
            },
        )
        # The common name should appear at least 3 times across S1 + S2 + S4.
        # Case-insensitive match because the wiki may not capitalise.
        n = out.lower().count("purple foxglove")
        assert n >= 3, f"common name appeared {n}x, expected >= 3: {out!r}"

    def test_repeats_scientific_name_at_least_twice(self) -> None:
        out = bec.build_lite_nameboost_answer(
            "purple foxglove", "Digitalis purpurea",
            {
                "scientific_name": "Digitalis purpurea",
                "wikipedia_summary":
                    "Digitalis purpurea is a toxic plant. It grows in Europe.",
                "gbif_distribution": "Europe",
            },
        )
        n = out.count("Digitalis purpurea")
        assert n >= 2, f"sci name appeared {n}x, expected >= 2: {out!r}"

    def test_caps_distribution_at_three_regions(self) -> None:
        out = bec.build_lite_nameboost_answer(
            "common dandelion", "Taraxacum officinale",
            {
                "scientific_name": "Taraxacum officinale",
                "wikipedia_summary": "A flowering plant.",
                "gbif_distribution": "; ".join(
                    f"Region{i}" for i in range(20)
                ),
            },
        )
        # Lite cap is 3 regions; ensure 4th onward not present and
        # the "and N more regions" tail is emitted.
        assert "Region0" in out and "Region1" in out and "Region2" in out
        assert "Region3" not in out
        assert "and 17 more regions" in out

    def test_caps_wiki_first_sentence(self) -> None:
        long_first = "x" * 500 + ". Second sentence here."
        out = bec.build_lite_nameboost_answer(
            "foo", "Foo bar",
            {"scientific_name": "Foo bar", "wikipedia_summary": long_first},
        )
        # Wiki cap is 180 chars; should NOT contain the full 500-char run
        # and should NOT contain the second sentence.
        assert "Second sentence here" not in out
        assert "x" * 500 not in out

    def test_falls_back_to_compact_when_enriched_is_none(self) -> None:
        out = bec.build_lite_nameboost_answer(
            "balsam fir", "Abies balsamea", enriched=None,
        )
        assert out == (
            "Looks like balsam fir to me. "
            "Abies balsamea, commonly called balsam fir."
        )

    def test_compact_fallback_skips_scientific_clause_when_unknown(
        self,
    ) -> None:
        out = bec.build_lite_nameboost_answer(
            "mystery plant", "(unknown)", enriched=None,
        )
        assert out == "Looks like mystery plant to me."

    def test_skips_s2_when_scientific_equals_common(self) -> None:
        # Degenerate case: avoid "Foo is also called Foo."
        out = bec.build_lite_nameboost_answer(
            "Foo bar", "Foo bar",
            {
                "scientific_name": "Foo bar",
                "wikipedia_summary": "A plant.",
                "gbif_distribution": "Europe",
            },
        )
        assert "is also called" not in out

    def test_drops_sentence_when_field_missing(self) -> None:
        # No wiki, no distribution → only S1 + S2 should appear.
        out = bec.build_lite_nameboost_answer(
            "foo", "Foo bar",
            {"scientific_name": "Foo bar"},
        )
        assert out == "Looks like foo to me. foo is also called Foo bar."

    def test_total_chars_in_target_band(self) -> None:
        # End-to-end sanity: a typical fully populated record lands in
        # the 200-500 char band (target: 250-400).
        out = bec.build_lite_nameboost_answer(
            "purple foxglove", "Digitalis purpurea",
            {
                "scientific_name": "Digitalis purpurea",
                "wikipedia_summary":
                    "Digitalis purpurea, the foxglove or common foxglove, "
                    "is a toxic species of flowering plant in the plantain "
                    "family Plantaginaceae, native to and widespread "
                    "throughout most of temperate Europe.",
                "gbif_distribution": "Europe; North America; Asia; Africa",
            },
        )
        assert 200 <= len(out) <= 500, (
            f"caption len {len(out)} outside target band: {out!r}"
        )


# ---------------------------------------------------------------------------
# _first_sentence
# ---------------------------------------------------------------------------

class TestFirstSentence:
    def test_returns_none_for_empty(self) -> None:
        assert bec._first_sentence(None, 100) is None
        assert bec._first_sentence("", 100) is None
        assert bec._first_sentence("   ", 100) is None

    def test_splits_on_period_space(self) -> None:
        out = bec._first_sentence("First. Second. Third.", 100)
        assert out == "First."

    def test_hard_caps_when_first_sentence_too_long(self) -> None:
        out = bec._first_sentence("a" * 200, 50)
        assert out is not None
        assert len(out) <= 50
        assert out.endswith("…")

    def test_returns_whole_string_when_no_boundary_under_cap(self) -> None:
        # No ". " before max_chars but the string itself is short
        # enough: return as-is.
        out = bec._first_sentence("short blurb no period", 100)
        assert out == "short blurb no period"


# ---------------------------------------------------------------------------
# _rebuild_row
# ---------------------------------------------------------------------------

class TestRebuildRow:
    def _row(self, slug: str) -> dict:
        return {
            "image": f"/abs/path/{slug}/img.jpg",
            "slug": slug,
            "species": "Foo bar",
            "family": "Fooaceae",
            "conversations": [
                {"role": "user", "content": "What plant is this?"},
                {"role": "assistant",
                 "content": "Looks like foo to me. Foo bar, commonly "
                            "called foo, is a plant species found in "
                            "North America."},
            ],
        }

    def test_preserves_image_slug_species_family_and_user_turn(self) -> None:
        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "accepted_scientific_name": "Foo bar",
                "wikipedia_summary": "A foo.",
            }
        }
        new_row, had_enrich = bec._rebuild_row(row, enriched, Counter())
        assert had_enrich
        # Image / slug / species / family preserved verbatim.
        for k in ("image", "slug", "species", "family"):
            assert new_row[k] == row[k]
        # User turn preserved verbatim.
        assert new_row["conversations"][0] == row["conversations"][0]
        # Assistant turn modified.
        assert new_row["conversations"][1]["role"] == "assistant"
        assert new_row["conversations"][1]["content"] != (
            row["conversations"][1]["content"]
        )

    def test_falls_back_when_slug_missing_from_enrichment(self) -> None:
        row = self._row("unknown_slug")
        new_row, had_enrich = bec._rebuild_row(row, {}, Counter())
        assert not had_enrich
        # Still produces a valid assistant content.
        assert new_row["conversations"][1]["content"].startswith(
            "Looks like unknown slug to me."
        )

    def test_hard_caps_oversized_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Shrink the cap so we can verify the truncation path on a
        # short string. The production constant is 1200 chars; that
        # would require building a giant fake enrichment.
        monkeypatch.setattr(bec, "MAX_CONTENT_CHARS", 80)

        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "accepted_scientific_name": "Foo bar",
                # Wikipedia summary much longer than the 80-char cap.
                "wikipedia_summary": "x" * 500,
            }
        }
        truncated: Counter = Counter()
        new_row, _ = bec._rebuild_row(row, enriched, truncated)
        content = new_row["conversations"][1]["content"]
        assert len(content) <= 80
        assert content.endswith("…")
        # First sentence (eval anchor) survives the cap.
        assert content.startswith("Looks like foo to me.")
        assert truncated["foo"] == 1

    def test_rejects_row_without_assistant_turn(self) -> None:
        row = {
            "image": "/abs/foo.jpg", "slug": "foo",
            "conversations": [{"role": "user", "content": "?"}],
        }
        with pytest.raises(ValueError, match="missing conversations"):
            bec._rebuild_row(row, {}, Counter())

    def test_rejects_unknown_mode(self) -> None:
        row = self._row("foo")
        with pytest.raises(ValueError, match="unknown caption mode"):
            bec._rebuild_row(row, {}, Counter(), mode="bogus_mode")

    def test_lite_nameboost_mode_routes_to_lite_builder(self) -> None:
        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "wikipedia_summary": "Foo bar is a plant. Second.",
                "gbif_distribution": "Europe; America; Asia; Africa",
            }
        }
        new_row, had_enrich = bec._rebuild_row(
            row, enriched, Counter(), mode="lite_nameboost",
        )
        assert had_enrich
        content = new_row["conversations"][1]["content"]
        assert content.startswith("Looks like foo to me.")
        # Lite mode's distinguishing sentence + range phrasing:
        assert "foo is also called Foo bar." in content
        assert "foo native range:" in content
        # Distribution cap (3 regions in lite) kicked in:
        assert "Africa" not in content
        assert "and 1 more regions" in content

    def test_lite_nameboost_hard_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Lite cap (default 500) must also be enforced.
        monkeypatch.setattr(bec, "MAX_CONTENT_CHARS_LITE", 60)
        # Bump wiki cap too so the wiki sentence stays long enough to
        # exceed the row cap.
        monkeypatch.setattr(bec, "MAX_WIKI_SENTENCE_CHARS_LITE", 200)

        row = self._row("foo")
        enriched = {
            "foo": {
                "scientific_name": "Foo bar",
                "wikipedia_summary": "x" * 200 + ".",
                "gbif_distribution": "Europe; America; Asia",
            },
        }
        truncated: Counter = Counter()
        new_row, _ = bec._rebuild_row(
            row, enriched, truncated, mode="lite_nameboost",
        )
        content = new_row["conversations"][1]["content"]
        assert len(content) <= 60
        assert content.endswith("…")
        assert content.startswith("Looks like foo to me.")
        assert truncated["foo"] == 1


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

def test_main_end_to_end_writes_jsonls_symlink_and_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = tmp_path / "in"
    output_root = tmp_path / "out"
    images_dir = input_root / "images_resized"
    images_dir.mkdir(parents=True)
    (images_dir / "img1.jpg").write_bytes(b"fake")

    # Two-row train, one-row val. test split intentionally absent so
    # the "missing split" path is exercised too.
    train_rows = [
        {
            "image": str(images_dir / "img1.jpg"),
            "slug": "foo",
            "species": "Foo bar",
            "family": "Fooaceae",
            "conversations": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant",
                 "content": "Looks like foo to me. Foo bar, commonly "
                            "called foo, is a plant species found in "
                            "North America."},
            ],
        },
        {
            "image": str(images_dir / "img1.jpg"),
            "slug": "unenriched_slug",
            "species": "Bar baz",
            "family": "(unknown)",
            "conversations": [
                {"role": "user", "content": "And this?"},
                {"role": "assistant",
                 "content": "Looks like unenriched_slug to me."},
            ],
        },
    ]
    val_rows = [train_rows[0]]
    (input_root / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )
    (input_root / "val.jsonl").write_text(
        json.dumps(val_rows[0]) + "\n"
    )

    enriched_path = tmp_path / "enriched.jsonl"
    enriched_path.write_text(json.dumps({
        "slug": "foo",
        "common_name": "foo",
        "scientific_name": "Foo bar",
        "accepted_scientific_name": "Foo bar",
        "common_names": "Foo; Fooey",
        "wikipedia_summary": "Foo is a plant.",
        "gbif_distribution": "North America (PRESENT); Maine; Vermont",
    }) + "\n")

    monkeypatch.setattr(sys, "argv", [
        "build_enriched_captions.py",
        "--input-root", str(input_root),
        "--enriched", str(enriched_path),
        "--output-root", str(output_root),
        "--log-level", "WARNING",
    ])
    bec.main()

    # Both rebuilt JSONLs exist.
    assert (output_root / "train.jsonl").exists()
    assert (output_root / "val.jsonl").exists()
    # test split skipped (warning logged, no file written).
    assert not (output_root / "test.jsonl").exists()

    # Images symlinked, not copied.
    sym = output_root / "images_resized"
    assert sym.is_symlink()
    assert sym.resolve() == images_dir.resolve()

    # Build report records the per-split counts and missing-enrichment
    # slug.
    report = json.loads((output_root / "build_report.json").read_text())
    assert report["enriched_unique_slugs"] == 1
    assert report["splits"]["train"]["n_rows"] == 2
    assert report["splits"]["train"]["n_rows_missing_enrichment"] == 1
    assert report["splits"]["train"]["missing_slugs_sample"] == [
        "unenriched_slug"
    ]
    assert "test" not in report["splits"]

    # Spot-check the rebuilt foo row: anchor preserved, wiki summary
    # included, GBIF distribution included.
    train_out = [
        json.loads(l)
        for l in (output_root / "train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    foo_row = next(r for r in train_out if r["slug"] == "foo")
    content = foo_row["conversations"][1]["content"]
    assert content.startswith("Looks like foo to me.")
    assert "Foo is a plant." in content
    assert "North America (PRESENT); Maine; Vermont" in content
    # Old "is a plant species found in North America." tail removed.
    assert "is a plant species found in North America" not in content


def test_main_end_to_end_lite_nameboost_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fixture as the full-mode end-to-end test, but with
    ``--mode lite_nameboost``. Verifies the report records the lite
    caps, the assistant content uses the lite template, and the
    average char count lands well below the full-mode mean.
    """
    input_root = tmp_path / "in"
    output_root = tmp_path / "out_lite"
    images_dir = input_root / "images_resized"
    images_dir.mkdir(parents=True)
    (images_dir / "img1.jpg").write_bytes(b"fake")

    train_rows = [
        {
            "image": str(images_dir / "img1.jpg"),
            "slug": "purple_foxglove",
            "species": "Digitalis purpurea",
            "family": "Plantaginaceae",
            "conversations": [
                {"role": "user", "content": "What is this?"},
                {"role": "assistant",
                 "content": "Looks like purple foxglove to me. ..."},
            ],
        },
    ]
    (input_root / "train.jsonl").write_text(
        "\n".join(json.dumps(r) for r in train_rows) + "\n"
    )

    enriched_path = tmp_path / "enriched.jsonl"
    enriched_path.write_text(json.dumps({
        "slug": "purple_foxglove",
        "common_name": "purple foxglove",
        "scientific_name": "Digitalis purpurea",
        "accepted_scientific_name": "Digitalis purpurea",
        "common_names": "Foxglove; Common Foxglove",
        "wikipedia_summary":
            "Digitalis purpurea, the foxglove or common foxglove, is a "
            "toxic species of flowering plant in the plantain family "
            "Plantaginaceae, native to and widespread throughout most of "
            "temperate Europe.",
        "gbif_distribution": "Europe; North America; Asia; Africa",
    }) + "\n")

    monkeypatch.setattr(sys, "argv", [
        "build_enriched_captions.py",
        "--input-root", str(input_root),
        "--enriched", str(enriched_path),
        "--output-root", str(output_root),
        "--splits", "train",
        "--mode", "lite_nameboost",
        "--log-level", "WARNING",
    ])
    bec.main()

    report = json.loads((output_root / "build_report.json").read_text())
    assert report["mode"] == "lite_nameboost"
    assert report["max_content_chars"] == bec.MAX_CONTENT_CHARS_LITE
    assert report["max_distribution_regions"] == bec.MAX_DISTRIBUTION_REGIONS_LITE
    assert report["max_wiki_sentence_chars"] == bec.MAX_WIKI_SENTENCE_CHARS_LITE
    # Lite mean should be well under 500 (typical: 250-400).
    assert report["splits"]["train"]["caption_chars_new"]["mean"] < 500

    train_out = [
        json.loads(l)
        for l in (output_root / "train.jsonl").read_text().splitlines()
        if l.strip()
    ]
    row = train_out[0]
    content = row["conversations"][1]["content"]
    # Eval anchor preserved.
    assert content.startswith("Looks like purple foxglove to me. ")
    # Lite template's distinguishing phrases.
    assert "purple foxglove is also called Digitalis purpurea." in content
    assert "purple foxglove native range:" in content
    # Distribution cap (3 regions, lite) fired:
    assert "Africa" not in content
    assert "and 1 more regions" in content
    # Common-name repeats >= 3.
    assert content.lower().count("purple foxglove") >= 3
