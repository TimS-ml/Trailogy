"""Tests for src.evaluate.extract_species.

Covers both the legacy Latin-binomial output format AND the new English
common name + description format introduced by prepare_plantnet_enriched.

Critical invariants:

  * No regression on Latin: trained-Gemma output like "This is Tsuga
    canadensis. ..." must still extract "tsuga canadensis".
  * English Title Case names extract cleanly: "This is Eastern Hemlock. ..."
    must produce "eastern hemlock" (NOT "eastern hemlock to me" or
    "you've spotted sugar maple").
  * All 8 ANSWER_TEMPLATES_RICH triggers fire correctly.
  * Sentence terminators are respected: `.`, `,`, `!`, `?`, `\n`,
    template-tail `" to me"`.
"""

from __future__ import annotations

import pytest

from src.evaluate import extract_species


# ---------------------------------------------------------------------------
# Latin binomial (legacy format) — must not regress
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("This is Lactuca virosa. You'll find it in many habitats.", "lactuca virosa"),
        ("That's Tsuga canadensis. It tends to grow in shade.", "tsuga canadensis"),
        ("Looks like Quercus alba to me. A great example of local flora.", "quercus alba"),
        ("You're looking at Acer rubrum. Nice find!", "acer rubrum"),
        ("This appears to be Pinus strobus. It's well-known.", "pinus strobus"),
    ],
)
def test_latin_binomial_extracts_correctly(text: str, expected: str) -> None:
    assert extract_species(text) == expected


# ---------------------------------------------------------------------------
# English common names (new enriched format)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # All 8 ANSWER_TEMPLATES_RICH triggers, each with a 2-word Title Case name.
        ("This is Eastern Hemlock. Tsuga canadensis is a coniferous tree.", "eastern hemlock"),
        ("You're looking at Sugar Maple. Acer saccharum is best known for maple syrup.", "sugar maple"),
        ("That's White Oak. Quercus alba is a hardwood.", "white oak"),
        ("Good eye — this is Bitter Lettuce. Lactuca virosa is often ingested for ...", "bitter lettuce"),
        ("That looks like Red Maple. Acer rubrum is one of the most common ...", "red maple"),
        ("Looks like American Beech to me. Fagus grandifolia, the American beech ...", "american beech"),
        ("You've spotted Tulip Tree. Liriodendron tulipifera grows to 35 m tall.", "tulip tree"),
        ("This appears to be Canadian Hemlock. It's well-known among botanists.", "canadian hemlock"),
    ],
)
def test_english_common_name_extracts_correctly(text: str, expected: str) -> None:
    assert extract_species(text) == expected


def test_single_word_english_common_name() -> None:
    assert extract_species("This is Hemlock. A coniferous tree.") == "hemlock"


def test_three_word_english_common_name() -> None:
    assert extract_species("This is Eastern White Oak. Range: North America.") == "eastern white oak"


# ---------------------------------------------------------------------------
# Terminator handling — these are the regressions the new regex fixes
# ---------------------------------------------------------------------------


def test_to_me_tail_does_not_leak_into_capture() -> None:
    # Bug #1 in the old regex: "Looks like X to me." captured "X to me".
    assert (
        extract_species("Looks like Eastern Hemlock to me. It tends to grow ...")
        == "eastern hemlock"
    )


def test_youve_spotted_trigger_is_recognized() -> None:
    # Bug #2: "You've spotted" was not in the trigger list, so the regex
    # fell through to the first-sentence fallback which returned the entire
    # phrase including the trigger.
    assert (
        extract_species("You've spotted Sugar Maple. The species is well-known.")
        == "sugar maple"
    )


def test_comma_terminator_stops_capture() -> None:
    assert extract_species("This is Eastern Hemlock, a coniferous tree.") == "eastern hemlock"


def test_newline_terminator_stops_capture() -> None:
    assert extract_species("This is Eastern Hemlock\nA second line.") == "eastern hemlock"


def test_question_mark_terminator() -> None:
    # Trigger fires, then `?` stops the capture (the model emits a rhetorical
    # follow-up after a confident identification).
    assert (
        extract_species("This is Eastern Hemlock? Yes — found in shade forests.")
        == "eastern hemlock"
    )


# ---------------------------------------------------------------------------
# Fallback patterns (bold / italic / first-sentence) — preserved behavior
# ---------------------------------------------------------------------------


def test_markdown_bold_fallback_still_works() -> None:
    # Verbose base-model output without a trigger phrase.
    assert extract_species("the plant is **Tsuga canadensis**") == "tsuga canadensis"


def test_italic_binomial_fallback() -> None:
    # Italic binomial with author abbreviation. extract_species lowercases
    # its output, so the author 'L.' comes back as 'l.'.
    assert extract_species("we have *Lactuca virosa L.* here") == "lactuca virosa l."


# ---------------------------------------------------------------------------
# Reference / prediction symmetry — same string from both sides matches
# ---------------------------------------------------------------------------


def test_species_match_symmetry_english() -> None:
    """If reference and prediction both come from ANSWER_TEMPLATES_RICH,
    extract_species must produce identical tokens on both sides so
    species_match works correctly."""
    reference = "You're looking at Eastern Hemlock. Tsuga canadensis is a coniferous tree."
    prediction = "Looks like Eastern Hemlock to me. It's one of the most common trees."
    assert extract_species(reference) == extract_species(prediction)


def test_species_match_symmetry_mixed_template_choice() -> None:
    # All 8 templates with the same species name should extract to the same token.
    species = "Sugar Maple"
    desc = "Acer saccharum is the primary source of maple syrup."
    from src.prepare_plantnet_enriched import ANSWER_TEMPLATES_RICH

    extracted = {
        extract_species(t.format(common=species, description=desc))
        for t in ANSWER_TEMPLATES_RICH
    }
    assert extracted == {"sugar maple"}
