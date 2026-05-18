"""VQAv2 unit tests — exercise the normalization + majority logic only.

The dataset-load and image-save paths are integration-tested via the
4090 smoke script; here we just verify the pure-Python helpers.
"""

from __future__ import annotations

from src.eval.vqav2 import _majority, _normalize


def test_normalize_strips_and_lowercases():
    assert _normalize("  Hello WORLD  ") == "hello world"
    assert _normalize("a\tb\nc") == "a b c"
    assert _normalize("") == ""
    assert _normalize(None) == ""


def test_majority_basic():
    assert _majority(["yes", "yes", "no"]) == "yes"
    assert _majority(["Yes", "yes", "YES"]) == "yes"
    assert _majority([]) == ""
    assert _majority(["one"]) == "one"


def test_majority_ties_pick_one_deterministically():
    # Counter.most_common picks insertion order on ties; we just
    # verify it returns *something* non-empty and from the input.
    result = _majority(["a", "b"])
    assert result in {"a", "b"}
